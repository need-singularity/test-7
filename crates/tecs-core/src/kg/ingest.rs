use super::types::*;
use anyhow::{Context, Result};
use log::{info, warn};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Wikidata5m 데이터셋 인제스터
///
/// 파일 포맷 (TSV):
///   - wikidata5m_transductive_train.txt: head_id \t relation_id \t tail_id
///   - wikidata5m_entity.txt: entity_id \t label
///   - wikidata5m_relation.txt: relation_id \t label
pub struct WikidataIngestor {
    /// base path to wikidata5m directory
    base_path: String,
    /// 관계 타입 필터 (None = 모두 포함)
    edge_filter: Option<Vec<RelationType>>,
    /// 최대 엔티티 수 제한 (0 = 무제한)
    max_entities: usize,
}

/// 인제스트 결과 통계
#[derive(Debug)]
pub struct IngestStats {
    pub total_entities: usize,
    pub total_triples: usize,
    pub hierarchical_edges: usize,
    pub cross_edges: usize,
    pub skipped_triples: usize,
    pub elapsed_secs: f64,
}

impl WikidataIngestor {
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            edge_filter: None,
            max_entities: 0,
        }
    }

    /// 특정 관계 타입만 포함하도록 필터 설정
    pub fn with_edge_filter(mut self, types: Vec<RelationType>) -> Self {
        self.edge_filter = Some(types);
        self
    }

    /// 최대 엔티티 수 제한
    pub fn with_max_entities(mut self, max: usize) -> Self {
        self.max_entities = max;
        self
    }

    /// Phase 1: 엔티티 레이블 로딩
    fn load_entity_labels(&self) -> Result<HashMap<String, String>> {
        let path = Path::new(&self.base_path).join("wikidata5m_entity.txt");
        let file = std::fs::File::open(&path)
            .with_context(|| format!("Cannot open entity file: {:?}", path))?;
        let reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer

        let mut labels = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            if parts.len() == 2 {
                labels.insert(parts[0].to_string(), parts[1].to_string());
            }
            if self.max_entities > 0 && labels.len() >= self.max_entities {
                break;
            }
        }

        info!("Loaded {} entity labels", labels.len());
        Ok(labels)
    }

    /// Phase 2: 트리플 로딩 → petgraph 구축
    pub fn ingest(&self) -> Result<(DiGraph<EntityNode, RelationEdge>, HashMap<String, NodeIndex>, IngestStats)> {
        let start = std::time::Instant::now();

        // 1) 레이블 로드
        let labels = self.load_entity_labels()?;

        // 2) 그래프 초기화 (대략적 크기 힌트)
        let estimated_nodes = if self.max_entities > 0 { self.max_entities } else { 5_000_000 };
        let mut graph = DiGraph::with_capacity(estimated_nodes, estimated_nodes * 4);
        let mut entity_index: HashMap<String, NodeIndex> = HashMap::with_capacity(estimated_nodes);

        // 노드를 가져오거나 생성하는 클로저
        let get_or_create_node = |
            graph: &mut DiGraph<EntityNode, RelationEdge>,
            index: &mut HashMap<String, NodeIndex>,
            labels: &HashMap<String, String>,
            entity_id: &str,
        | -> NodeIndex {
            if let Some(&idx) = index.get(entity_id) {
                return idx;
            }
            let label = labels.get(entity_id)
                .cloned()
                .unwrap_or_else(|| entity_id.to_string());
            let node = EntityNode {
                wikidata_id: entity_id.to_string(),
                label,
                depth: None,
                degree: 0,
            };
            let idx = graph.add_node(node);
            index.insert(entity_id.to_string(), idx);
            idx
        };

        // 3) 트리플 파일 스트리밍 파싱
        let triples_path = Path::new(&self.base_path).join("wikidata5m_transductive_train.txt");
        let file = std::fs::File::open(&triples_path)
            .with_context(|| format!("Cannot open triples file: {:?}", triples_path))?;
        let reader = BufReader::with_capacity(4 * 1024 * 1024, file); // 4MB buffer

        let mut total_triples = 0usize;
        let mut hier_edges = 0usize;
        let mut cross_edges = 0usize;
        let mut skipped = 0usize;

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                skipped += 1;
                continue;
            }

            let (head_id, rel_id, tail_id) = (parts[0], parts[1], parts[2]);
            let rel_type = RelationType::from_property_id(rel_id);

            // 엣지 필터 적용
            if let Some(ref filter) = self.edge_filter {
                if !filter.contains(&rel_type) && !matches!(rel_type, RelationType::Other(_)) {
                    skipped += 1;
                    continue;
                }
            }

            // 최대 엔티티 수 체크
            if self.max_entities > 0 && entity_index.len() >= self.max_entities {
                if !entity_index.contains_key(head_id) || !entity_index.contains_key(tail_id) {
                    skipped += 1;
                    continue;
                }
            }

            let head_idx = get_or_create_node(&mut graph, &mut entity_index, &labels, head_id);
            let tail_idx = get_or_create_node(&mut graph, &mut entity_index, &labels, tail_id);

            let edge = RelationEdge {
                property_id: rel_id.to_string(),
                relation_type: rel_type,
                priority: rel_type.priority_score(),
            };

            graph.add_edge(head_idx, tail_idx, edge);

            // degree 업데이트
            graph[head_idx].degree += 1;
            graph[tail_idx].degree += 1;

            if rel_type.is_hierarchical() {
                hier_edges += 1;
            } else {
                cross_edges += 1;
            }

            total_triples += 1;

            if total_triples % 1_000_000 == 0 {
                info!("Processed {} triples, {} entities...", total_triples, entity_index.len());
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let stats = IngestStats {
            total_entities: entity_index.len(),
            total_triples,
            hierarchical_edges: hier_edges,
            cross_edges,
            skipped_triples: skipped,
            elapsed_secs: elapsed,
        };

        info!("Ingestion complete: {:?}", stats);
        Ok((graph, entity_index, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_data(dir: &Path) {
        // Entity labels
        let mut f = std::fs::File::create(dir.join("wikidata5m_entity.txt")).unwrap();
        writeln!(f, "Q413\tPhysics").unwrap();
        writeln!(f, "Q11023\tEngineering").unwrap();
        writeln!(f, "Q2329\tChemistry").unwrap();
        writeln!(f, "Q395\tMathematics").unwrap();
        writeln!(f, "Q944\tQuantum mechanics").unwrap();

        // Triples
        let mut f = std::fs::File::create(dir.join("wikidata5m_transductive_train.txt")).unwrap();
        writeln!(f, "Q944\tP279\tQ413").unwrap();       // QM subclass-of Physics
        writeln!(f, "Q413\tP279\tQ395").unwrap();       // Physics subclass-of Math (approximate)
        writeln!(f, "Q2329\tP279\tQ413").unwrap();      // Chemistry subclass-of Physics (approximate)
        writeln!(f, "Q11023\tP2283\tQ413").unwrap();    // Engineering uses Physics
        writeln!(f, "Q944\tP737\tQ2329").unwrap();      // QM influenced-by Chemistry
    }

    #[test]
    fn test_basic_ingestion() {
        let dir = TempDir::new().unwrap();
        create_test_data(dir.path());

        let ingestor = WikidataIngestor::new(dir.path().to_str().unwrap());
        let (graph, index, stats) = ingestor.ingest().unwrap();

        assert_eq!(stats.total_entities, 5);
        assert_eq!(stats.total_triples, 5);
        assert_eq!(stats.hierarchical_edges, 3); // 3x P279
        assert_eq!(stats.cross_edges, 2);        // P2283 + P737
        assert!(index.contains_key("Q413"));
    }

    #[test]
    fn test_filtered_ingestion() {
        let dir = TempDir::new().unwrap();
        create_test_data(dir.path());

        let ingestor = WikidataIngestor::new(dir.path().to_str().unwrap())
            .with_edge_filter(vec![RelationType::SubclassOf, RelationType::InstanceOf]);
        let (_, _, stats) = ingestor.ingest().unwrap();

        assert_eq!(stats.hierarchical_edges, 3);
        // Other edges still loaded because of Other(_) fallthrough
    }
}
