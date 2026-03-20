use super::types::*;
use anyhow::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

/// 메인 Knowledge Graph 스토어
#[derive(Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// petgraph directed graph
    pub graph: DiGraph<EntityNode, RelationEdge>,
    /// Wikidata ID → NodeIndex 매핑
    pub entity_index: HashMap<String, NodeIndex>,
    /// 계층 루트 노드 (가장 일반적인 개념)
    pub root: Option<NodeIndex>,
}

/// 서브그래프 추출 결과
#[derive(Debug, Clone)]
pub struct SubGraph {
    /// 포함된 노드들
    pub nodes: Vec<NodeIndex>,
    /// 노드 → 로컬 인덱스 매핑
    pub node_to_local: HashMap<NodeIndex, usize>,
    /// 엣지 목록 (local_src, local_dst, relation_type)
    pub edges: Vec<(usize, usize, RelationType)>,
}

impl KnowledgeGraph {
    /// DiGraph + entity_index로부터 KG 구성
    pub fn new(
        graph: DiGraph<EntityNode, RelationEdge>,
        entity_index: HashMap<String, NodeIndex>,
    ) -> Self {
        let mut kg = Self {
            graph,
            entity_index,
            root: None,
        };
        kg.root = Some(kg.find_best_root());
        kg
    }

    /// 가장 연결이 많은 노드를 루트로 선택
    fn find_best_root(&self) -> NodeIndex {
        self.graph.node_indices()
            .max_by_key(|&idx| self.graph[idx].degree)
            .unwrap_or(NodeIndex::new(0))
    }

    /// Wikidata ID로 NodeIndex 찾기
    pub fn resolve(&self, wikidata_id: &str) -> Option<NodeIndex> {
        self.entity_index.get(wikidata_id).copied()
    }

    /// 여러 Wikidata ID를 한번에 resolve
    pub fn resolve_many(&self, ids: &[&str]) -> Vec<Option<NodeIndex>> {
        ids.iter().map(|id| self.resolve(id)).collect()
    }

    /// BFS로 k-hop 서브그래프 추출
    pub fn extract_subgraph(
        &self,
        seeds: &[NodeIndex],
        max_hops: u32,
        max_nodes: usize,
        min_priority: u8,
    ) -> SubGraph {
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, u32)> = VecDeque::new();
        let mut nodes: Vec<NodeIndex> = Vec::new();

        // seed 노드들 큐에 넣기
        for &seed in seeds {
            if visited.insert(seed) {
                queue.push_back((seed, 0));
                nodes.push(seed);
            }
        }

        // BFS 탐색
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_hops || nodes.len() >= max_nodes {
                continue;
            }

            // outgoing edges
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                let weight = edge.weight();
                if weight.priority < min_priority {
                    continue;
                }
                let target = edge.target();
                if visited.insert(target) {
                    queue.push_back((target, depth + 1));
                    nodes.push(target);
                    if nodes.len() >= max_nodes {
                        break;
                    }
                }
            }

            if nodes.len() >= max_nodes {
                break;
            }

            // incoming edges (양방향 탐색)
            for edge in self.graph.edges_directed(node, Direction::Incoming) {
                let weight = edge.weight();
                if weight.priority < min_priority {
                    continue;
                }
                let source = edge.source();
                if visited.insert(source) {
                    queue.push_back((source, depth + 1));
                    nodes.push(source);
                    if nodes.len() >= max_nodes {
                        break;
                    }
                }
            }
        }

        // 로컬 인덱스 매핑 + 서브그래프 엣지 추출
        let node_to_local: HashMap<NodeIndex, usize> = nodes.iter()
            .enumerate()
            .map(|(i, &n)| (n, i))
            .collect();

        let mut edges = Vec::new();
        for &node in &nodes {
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                if let Some(&local_dst) = node_to_local.get(&edge.target()) {
                    let local_src = node_to_local[&node];
                    edges.push((local_src, local_dst, edge.weight().relation_type));
                }
            }
        }

        SubGraph { nodes, node_to_local, edges }
    }

    /// Spanning tree 추출 (계층 엣지만 사용)
    pub fn extract_hierarchy_tree(&self, root: NodeIndex) -> DiGraph<NodeIndex, ()> {
        let mut tree = DiGraph::new();
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();
        let mut tree_nodes: HashMap<NodeIndex, petgraph::graph::NodeIndex> = HashMap::new();

        // 루트 추가
        let tree_root = tree.add_node(root);
        tree_nodes.insert(root, tree_root);
        visited.insert(root);
        queue.push_back(root);

        while let Some(node) = queue.pop_front() {
            // incoming hierarchical edges: "X subclass-of node" → X는 node의 자식
            for edge in self.graph.edges_directed(node, Direction::Incoming) {
                if !edge.weight().relation_type.is_hierarchical() {
                    continue;
                }
                let child = edge.source();
                if visited.insert(child) {
                    let tree_child = tree.add_node(child);
                    tree_nodes.insert(child, tree_child);
                    tree.add_edge(tree_nodes[&node], tree_child, ());
                    queue.push_back(child);
                }
            }

            // outgoing hierarchical edges: "node subclass-of Y" → Y는 node의 부모
            // (역방향이므로 tree에서는 Y의 자식으로 node를 넣되, 이미 처리된 경우 skip)
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                if !edge.weight().relation_type.is_hierarchical() {
                    continue;
                }
                let parent = edge.target();
                if visited.insert(parent) {
                    let tree_parent = tree.add_node(parent);
                    tree_nodes.insert(parent, tree_parent);
                    // tree에서 parent → node 방향
                    tree.add_edge(tree_parent, tree_nodes[&node], ());
                    queue.push_back(parent);
                }
            }
        }

        tree
    }

    /// 캐시를 위한 바이너리 직렬화
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// 캐시에서 로드
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let kg: Self = bincode::deserialize(&bytes)?;
        Ok(kg)
    }

    /// 노드 수
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// 엣지 수
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_kg() -> KnowledgeGraph {
        let mut graph = DiGraph::new();
        let mut index = HashMap::new();

        let science = graph.add_node(EntityNode {
            wikidata_id: "Q336".into(), label: "Science".into(), depth: None, degree: 0,
        });
        let physics = graph.add_node(EntityNode {
            wikidata_id: "Q413".into(), label: "Physics".into(), depth: None, degree: 0,
        });
        let qm = graph.add_node(EntityNode {
            wikidata_id: "Q944".into(), label: "Quantum mechanics".into(), depth: None, degree: 0,
        });

        graph.add_edge(physics, science, RelationEdge {
            property_id: "P279".into(),
            relation_type: RelationType::SubclassOf,
            priority: 10,
        });
        graph.add_edge(qm, physics, RelationEdge {
            property_id: "P279".into(),
            relation_type: RelationType::SubclassOf,
            priority: 10,
        });

        index.insert("Q336".into(), science);
        index.insert("Q413".into(), physics);
        index.insert("Q944".into(), qm);

        KnowledgeGraph::new(graph, index)
    }

    #[test]
    fn test_subgraph_extraction() {
        let kg = build_test_kg();
        let physics = kg.resolve("Q413").unwrap();
        let sub = kg.extract_subgraph(&[physics], 2, 100, 1);
        assert_eq!(sub.nodes.len(), 3); // Science, Physics, QM
    }

    #[test]
    fn test_hierarchy_tree() {
        let kg = build_test_kg();
        let science = kg.resolve("Q336").unwrap();
        let tree = kg.extract_hierarchy_tree(science);
        assert!(tree.node_count() >= 2);
    }
}
