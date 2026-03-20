use crate::hyperbolic::HyperbolicEmbedding;
use nalgebra::DMatrix;
use petgraph::graph::NodeIndex;
use rayon::prelude::*;

/// 쌍곡 거리 매트릭스 생성기
pub struct DistanceMatrixBuilder;

impl DistanceMatrixBuilder {
    /// 노드 목록으로부터 쌍곡 거리 매트릭스 생성
    /// 상삼각만 계산하고 대칭 복사 (n*(n-1)/2 연산)
    /// rayon으로 행 단위 병렬화
    pub fn build(
        embedding: &HyperbolicEmbedding,
        nodes: &[NodeIndex],
    ) -> DMatrix<f64> {
        let n = nodes.len();
        let mut dm = DMatrix::zeros(n, n);

        // 각 노드의 좌표를 미리 추출 (반복 접근 최적화)
        let coords: Vec<_> = nodes.iter()
            .map(|node| {
                let row = embedding.node_to_row[node];
                embedding.coords.row(row).to_owned()
            })
            .collect();

        // 행 단위 병렬 계산
        let upper_triangle: Vec<Vec<(usize, usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row_entries = Vec::with_capacity(n - i - 1);
                for j in (i + 1)..n {
                    let d = embedding.ball.distance(
                        &coords[i].view(),
                        &coords[j].view(),
                    );
                    row_entries.push((i, j, d));
                }
                row_entries
            })
            .collect();

        // 결과를 매트릭스에 기록 (대칭)
        for row_entries in upper_triangle {
            for (i, j, d) in row_entries {
                dm[(i, j)] = d;
                dm[(j, i)] = d;
            }
        }

        dm
    }
}

/// Compute graph shortest-path distance matrix (Floyd-Warshall)
/// edges: list of (src, dst) pairs (0-indexed, undirected)
/// n: number of nodes
pub fn graph_distance_matrix(edges: &[(usize, usize)], n: usize) -> DMatrix<f64> {
    let mut dm = DMatrix::from_element(n, n, f64::INFINITY);

    // Self-distance = 0
    for i in 0..n {
        dm[(i, i)] = 0.0;
    }

    // Direct edges = distance 1.0
    for &(u, v) in edges {
        dm[(u, v)] = 1.0;
        dm[(v, u)] = 1.0;
    }

    // Floyd-Warshall
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through_k = dm[(i, k)] + dm[(k, j)];
                if through_k < dm[(i, j)] {
                    dm[(i, j)] = through_k;
                }
            }
        }
    }

    dm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix_symmetry() {
        // symmetry와 triangle inequality는 hyperball 자체에서 보장하지만
        // 우리 빌더가 올바르게 조립하는지 확인
        // (실제 임베딩 필요 — integration test에서)
    }

    #[test]
    fn test_graph_distance_matrix_4node_cycle() {
        // 4-node cycle: A-B, B-C, C-D, D-A
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let dm = graph_distance_matrix(&edges, 4);

        // Self-distances = 0
        for i in 0..4 {
            assert_eq!(dm[(i, i)], 0.0, "Self-distance should be 0");
        }

        // Adjacent nodes = distance 1
        assert_eq!(dm[(0, 1)], 1.0, "A-B adjacent");
        assert_eq!(dm[(1, 2)], 1.0, "B-C adjacent");
        assert_eq!(dm[(2, 3)], 1.0, "C-D adjacent");
        assert_eq!(dm[(3, 0)], 1.0, "D-A adjacent");

        // Opposite nodes = distance 2
        assert_eq!(dm[(0, 2)], 2.0, "A-C opposite");
        assert_eq!(dm[(1, 3)], 2.0, "B-D opposite");

        // Symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(dm[(i, j)], dm[(j, i)], "Symmetry check");
            }
        }
    }
}
