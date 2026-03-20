use crate::kg::{KnowledgeGraph, RelationType};
use anyhow::{Context, Result};
use hyperball::PoincareBall;
use log::info;
use ndarray::{Array1, Array2};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use skel::Manifold;
use std::collections::{HashMap, HashSet, VecDeque};

/// Sarkar 알고리즘으로 생성된 쌍곡 임베딩
pub struct HyperbolicEmbedding {
    /// Poincaré ball 모델 (curvature c)
    pub ball: PoincareBall<f64>,
    /// 좌표 행렬 (n_nodes × dim)
    pub coords: Array2<f64>,
    /// NodeIndex → 행렬 행 번호
    pub node_to_row: HashMap<NodeIndex, usize>,
    /// 행 번호 → NodeIndex (역매핑)
    pub row_to_node: Vec<NodeIndex>,
    /// 임베딩 차원
    pub dim: usize,
    /// 스케일 파라미터
    pub tau: f64,
}

/// Sarkar 임베딩 설정
pub struct SarkarConfig {
    /// 임베딩 차원 (2-16, 보통 2-8이면 충분)
    pub dim: usize,
    /// 노드 간 쌍곡 거리 스케일 (0.1-0.5 권장)
    pub tau: f64,
    /// 경계 안전 거리 (||x|| < 1-epsilon)
    pub boundary_eps: f64,
    /// curvature (보통 1.0)
    pub curvature: f64,
}

impl Default for SarkarConfig {
    fn default() -> Self {
        Self {
            dim: 8,
            tau: 0.3,
            boundary_eps: 1e-5,
            curvature: 1.0,
        }
    }
}

impl HyperbolicEmbedding {
    /// Sarkar 알고리즘: 지식 그래프 → Poincaré ball 임베딩
    ///
    /// 1. KG에서 계층 엣지로 spanning tree 추출 (BFS)
    /// 2. 루트를 원점에 배치
    /// 3. 각 자식을 부모로부터 tau 거리만큼 떨어진 위치에 exp_map으로 배치
    /// 4. 형제 노드는 등간격 각도로 분포
    ///
    /// 시간 복잡도: O(n), GPU 불필요
    pub fn sarkar_embed(kg: &KnowledgeGraph, config: &SarkarConfig) -> Result<Self> {
        let ball = PoincareBall::new(config.curvature);
        let root = kg.root.context("KG has no root node")?;

        // Phase 1: BFS로 spanning tree 구축 + depth 계산
        let (parent_map, children_map, depth_map) = Self::build_tree_bfs(kg, root);

        let n = parent_map.len() + 1; // +1 for root
        info!("Sarkar embedding: {} nodes, dim={}", n, config.dim);

        let mut coords = Array2::zeros((n, config.dim));
        let mut node_to_row: HashMap<NodeIndex, usize> = HashMap::with_capacity(n);
        let mut row_to_node: Vec<NodeIndex> = Vec::with_capacity(n);

        // Phase 2: 루트를 원점에 배치
        node_to_row.insert(root, 0);
        row_to_node.push(root);
        // coords[0] = [0, 0, ..., 0] — 이미 zero

        // Phase 3: BFS 순서로 자식 노드 배치
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();
        queue.push_back(root);
        let mut row_counter: usize = 1;

        while let Some(parent) = queue.pop_front() {
            let children = match children_map.get(&parent) {
                Some(c) => c,
                None => continue,
            };

            let n_children = children.len();
            if n_children == 0 {
                continue;
            }

            let parent_row = node_to_row[&parent];
            let parent_coord = coords.row(parent_row).to_owned();

            // 부모의 depth로 tau 스케일링 (깊을수록 밀집)
            let parent_depth = depth_map.get(&parent).copied().unwrap_or(0) as f64;
            let scaled_tau = config.tau * (0.95_f64).powf(parent_depth); // 점진적 축소

            for (i, child) in children.iter().enumerate() {
                // 각도: 형제들 사이에 등간격 분배
                // dim >= 2에서 처음 두 차원을 각도로 사용
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_children as f64);

                // tangent vector at parent (dim차원)
                let mut tangent = Array1::zeros(config.dim);
                tangent[0] = scaled_tau * angle.cos();
                if config.dim > 1 {
                    tangent[1] = scaled_tau * angle.sin();
                }

                // dim > 2인 경우: 추가 차원에 depth 기반 정보 인코딩
                if config.dim > 2 {
                    let child_depth = depth_map.get(child).copied().unwrap_or(0) as f64;
                    let max_depth = depth_map.values().max().copied().unwrap_or(10) as f64;
                    tangent[2] = scaled_tau * 0.1 * (child_depth / max_depth);
                }

                // exp_map: tangent space → Poincaré ball
                let child_coord = Self::exp_map_safe(
                    &ball,
                    &parent_coord,
                    &tangent,
                    config.boundary_eps,
                );

                // 좌표 저장
                let row = row_counter;
                row_counter += 1;

                if row < n {
                    coords.row_mut(row).assign(&child_coord);
                    node_to_row.insert(*child, row);
                    row_to_node.push(*child);
                    queue.push_back(*child);
                }
            }
        }

        info!(
            "Sarkar embedding complete: {} nodes embedded, max_norm={:.6}",
            row_counter,
            (0..row_counter).map(|r| {
                let row = coords.row(r);
                row.dot(&row).sqrt()
            }).fold(0.0_f64, f64::max)
        );

        Ok(Self {
            ball,
            coords,
            node_to_row,
            row_to_node,
            dim: config.dim,
            tau: config.tau,
        })
    }

    /// BFS로 spanning tree 구축
    fn build_tree_bfs(
        kg: &KnowledgeGraph,
        root: NodeIndex,
    ) -> (
        HashMap<NodeIndex, NodeIndex>,        // child → parent
        HashMap<NodeIndex, Vec<NodeIndex>>,   // parent → children
        HashMap<NodeIndex, u32>,               // node → depth
    ) {
        let mut parent_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut children_map: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
        let mut depth_map: HashMap<NodeIndex, u32> = HashMap::new();
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, u32)> = VecDeque::new();

        visited.insert(root);
        depth_map.insert(root, 0);
        queue.push_back((root, 0));

        while let Some((node, depth)) = queue.pop_front() {
            // 계층 엣지의 자식들 탐색
            // "X --P279--> node" means X is a subclass of node → X is child
            for edge in kg.graph.edges_directed(node, Direction::Incoming) {
                if !edge.weight().relation_type.is_hierarchical() {
                    continue;
                }
                let child = edge.source();
                if visited.insert(child) {
                    parent_map.insert(child, node);
                    children_map.entry(node).or_default().push(child);
                    depth_map.insert(child, depth + 1);
                    queue.push_back((child, depth + 1));
                }
            }

            // "node --P279--> Y" means node is a subclass of Y
            // 만약 Y가 아직 방문되지 않았다면, Y를 루트 쪽으로 탐색
            for edge in kg.graph.edges_directed(node, Direction::Outgoing) {
                if !edge.weight().relation_type.is_hierarchical() {
                    continue;
                }
                let parent_candidate = edge.target();
                if visited.insert(parent_candidate) {
                    // 이 경우 node의 부모로 설정하지 않고,
                    // parent_candidate를 별도 subtree로 추가
                    children_map.entry(node).or_default().push(parent_candidate);
                    parent_map.insert(parent_candidate, node);
                    depth_map.insert(parent_candidate, depth + 1);
                    queue.push_back((parent_candidate, depth + 1));
                }
            }
        }

        (parent_map, children_map, depth_map)
    }

    /// 안전한 exp_map (경계 클리핑 포함)
    fn exp_map_safe(
        ball: &PoincareBall<f64>,
        base: &Array1<f64>,
        tangent: &Array1<f64>,
        boundary_eps: f64,
    ) -> Array1<f64> {
        let result = ball.exp_map(&base.view(), &tangent.view());

        // 경계 안전성: ||x|| < 1 - eps
        let norm_sq: f64 = result.iter().map(|x| x * x).sum();
        let max_norm = 1.0 - boundary_eps;
        if norm_sq >= max_norm * max_norm {
            let norm = norm_sq.sqrt();
            let scale = max_norm / norm;
            return result.mapv(|x| x * scale);
        }

        result
    }

    /// 특정 노드의 쌍곡 좌표 조회
    pub fn get_coords(&self, node: NodeIndex) -> Option<Array1<f64>> {
        self.node_to_row.get(&node).map(|&row| {
            self.coords.row(row).to_owned()
        })
    }

    /// 임베딩된 노드 수
    pub fn node_count(&self) -> usize {
        self.node_to_row.len()
    }

    /// 두 노드 사이 쌍곡 거리
    pub fn hyperbolic_distance(&self, a: NodeIndex, b: NodeIndex) -> Option<f64> {
        let ra = self.node_to_row.get(&a)?;
        let rb = self.node_to_row.get(&b)?;
        Some(self.ball.distance(
            &self.coords.row(*ra),
            &self.coords.row(*rb),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::graph::KnowledgeGraph;
    use petgraph::graph::DiGraph;

    #[test]
    fn test_sarkar_basic() {
        // 간단한 3-레벨 트리: Science → Physics → QM
        let mut graph = DiGraph::new();
        let mut index = HashMap::new();

        let science = graph.add_node(crate::kg::EntityNode {
            wikidata_id: "Q336".into(), label: "Science".into(),
            depth: None, degree: 2,
        });
        let physics = graph.add_node(crate::kg::EntityNode {
            wikidata_id: "Q413".into(), label: "Physics".into(),
            depth: None, degree: 2,
        });
        let qm = graph.add_node(crate::kg::EntityNode {
            wikidata_id: "Q944".into(), label: "QM".into(),
            depth: None, degree: 1,
        });

        graph.add_edge(physics, science, crate::kg::RelationEdge {
            property_id: "P279".into(),
            relation_type: RelationType::SubclassOf,
            priority: 10,
        });
        graph.add_edge(qm, physics, crate::kg::RelationEdge {
            property_id: "P279".into(),
            relation_type: RelationType::SubclassOf,
            priority: 10,
        });

        index.insert("Q336".into(), science);
        index.insert("Q413".into(), physics);
        index.insert("Q944".into(), qm);

        let kg = KnowledgeGraph::new(graph, index);
        let config = SarkarConfig { dim: 2, tau: 0.3, ..Default::default() };

        let emb = HyperbolicEmbedding::sarkar_embed(&kg, &config).unwrap();

        // All 3 nodes should be embedded
        assert_eq!(emb.node_count(), 3);

        // All coordinates should be inside the Poincaré ball (||x|| < 1)
        for node in [science, physics, qm] {
            let coord = emb.get_coords(node).unwrap();
            let norm: f64 = coord.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1.0, "Node should be inside Poincaré ball, got norm {}", norm);
        }

        // Hyperbolic distances should be positive and finite
        let d_sp = emb.hyperbolic_distance(science, physics).unwrap();
        let d_sq = emb.hyperbolic_distance(science, qm).unwrap();
        let d_pq = emb.hyperbolic_distance(physics, qm).unwrap();
        assert!(d_sp > 0.0 && d_sp.is_finite(), "d(Science,Physics) should be positive finite");
        assert!(d_sq > 0.0 && d_sq.is_finite(), "d(Science,QM) should be positive finite");
        assert!(d_pq > 0.0 && d_pq.is_finite(), "d(Physics,QM) should be positive finite");
    }
}
