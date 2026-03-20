use crate::hyperbolic::HyperbolicEmbedding;
use log::debug;
use petgraph::graph::NodeIndex;
use rayon::prelude::*;

/// ε-neighborhood 프루닝 결과
#[derive(Debug, Clone)]
pub struct PruneResult {
    /// 프루닝된 노드들 (거리순 정렬)
    pub nodes: Vec<NodeIndex>,
    /// 각 노드의 쌍곡 거리
    pub distances: Vec<f64>,
    /// 프루닝 전 총 노드 수
    pub total_candidates: usize,
}

/// 기하학적 프루너
pub struct GeometricPruner {
    /// 최대 노드 수
    pub max_nodes: usize,
    /// 최소 노드 수 (이보다 적으면 ε 자동 확장)
    pub min_nodes: usize,
}

impl Default for GeometricPruner {
    fn default() -> Self {
        Self {
            max_nodes: 3000,
            min_nodes: 10,
        }
    }
}

impl GeometricPruner {
    pub fn new(max_nodes: usize, min_nodes: usize) -> Self {
        Self { max_nodes, min_nodes }
    }

    /// ε-neighborhood: query 노드로부터 쌍곡 거리 ε 이내의 노드 추출
    /// rayon으로 병렬 계산
    pub fn prune(
        &self,
        embedding: &HyperbolicEmbedding,
        query: NodeIndex,
        epsilon: f64,
    ) -> PruneResult {
        let query_row = match embedding.node_to_row.get(&query) {
            Some(&r) => r,
            None => return PruneResult {
                nodes: vec![], distances: vec![], total_candidates: 0,
            },
        };
        let query_coord = embedding.coords.row(query_row);

        let total_candidates = embedding.node_count();

        // rayon 병렬: 모든 노드와의 쌍곡 거리 계산
        let mut candidates: Vec<(NodeIndex, f64)> = embedding.row_to_node
            .par_iter()
            .enumerate()
            .filter_map(|(row, &node)| {
                if row == query_row {
                    return Some((node, 0.0)); // self-distance = 0
                }
                let d = embedding.ball.distance(
                    &query_coord,
                    &embedding.coords.row(row),
                );
                if d <= epsilon {
                    Some((node, d))
                } else {
                    None
                }
            })
            .collect();

        // 거리순 정렬
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // max_nodes 제한
        candidates.truncate(self.max_nodes);

        debug!(
            "Pruned {}/{} nodes within ε={}",
            candidates.len(), total_candidates, epsilon
        );

        let (nodes, distances): (Vec<_>, Vec<_>) = candidates.into_iter().unzip();

        PruneResult {
            nodes,
            distances,
            total_candidates,
        }
    }
}
