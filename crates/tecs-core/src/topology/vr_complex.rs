use lophat::columns::VecColumn;
use nalgebra::DMatrix;

/// Vietoris-Rips complex 빌더
/// 거리 매트릭스로부터 filtration 생성
pub struct VietorisRipsBuilder {
    /// 최대 simplex 차원 (1 = edges만, 2 = triangles까지)
    pub max_dim: usize,
    /// 최대 filtration 값
    pub max_radius: f64,
}

/// filtration 내 simplex
#[derive(Debug, Clone)]
pub struct FilteredSimplex {
    /// simplex 차원 (0=vertex, 1=edge, 2=triangle)
    pub dimension: usize,
    /// simplex를 구성하는 vertex 인덱스들
    pub vertices: Vec<usize>,
    /// filtration 값 (이 simplex가 나타나는 거리)
    pub filtration_value: f64,
    /// boundary의 simplex 인덱스들 (lophat 입력용)
    pub boundary_indices: Vec<usize>,
}

impl VietorisRipsBuilder {
    pub fn new(max_dim: usize, max_radius: f64) -> Self {
        Self { max_dim, max_radius }
    }

    /// 거리 매트릭스로부터 Vietoris-Rips filtration 생성
    ///
    /// 반환: filtration 순서로 정렬된 simplex 목록
    pub fn build_filtration(&self, dm: &DMatrix<f64>) -> Vec<FilteredSimplex> {
        let n = dm.nrows();
        let mut simplices: Vec<FilteredSimplex> = Vec::new();

        // Dim 0: 모든 vertex (filtration value = 0)
        for i in 0..n {
            simplices.push(FilteredSimplex {
                dimension: 0,
                vertices: vec![i],
                filtration_value: 0.0,
                boundary_indices: vec![],
            });
        }

        if self.max_dim < 1 {
            return simplices;
        }

        // Dim 1: edges (filtration value = distance)
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dm[(i, j)];
                if d <= self.max_radius {
                    edges.push((i, j, d));
                }
            }
        }
        // filtration 순서로 정렬
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // edge → simplex index 매핑
        let mut edge_index: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();

        for (i, j, d) in &edges {
            let idx = simplices.len();
            edge_index.insert((*i, *j), idx);
            simplices.push(FilteredSimplex {
                dimension: 1,
                vertices: vec![*i, *j],
                filtration_value: *d,
                boundary_indices: vec![*i, *j], // boundary = two vertices
            });
        }

        if self.max_dim < 2 {
            return simplices;
        }

        // Dim 2: triangles (filtration value = max edge distance)
        for i in 0..n {
            for j in (i + 1)..n {
                if dm[(i, j)] > self.max_radius { continue; }
                for k in (j + 1)..n {
                    if dm[(i, k)] > self.max_radius || dm[(j, k)] > self.max_radius {
                        continue;
                    }
                    // triangle {i, j, k}의 filtration = max of 3 edges
                    let filt = dm[(i, j)].max(dm[(i, k)]).max(dm[(j, k)]);
                    if filt > self.max_radius { continue; }

                    // boundary = 3 edges: {j,k}, {i,k}, {i,j}
                    let b_jk = edge_index.get(&(j, k)).copied();
                    let b_ik = edge_index.get(&(i, k)).copied();
                    let b_ij = edge_index.get(&(i, j)).copied();

                    if let (Some(e_jk), Some(e_ik), Some(e_ij)) = (b_jk, b_ik, b_ij) {
                        let mut boundary = vec![e_ij, e_ik, e_jk];
                        boundary.sort();
                        simplices.push(FilteredSimplex {
                            dimension: 2,
                            vertices: vec![i, j, k],
                            filtration_value: filt,
                            boundary_indices: boundary,
                        });
                    }
                }
            }
        }

        // 전체를 (dimension, filtration_value) 순으로 정렬
        simplices.sort_by(|a, b| {
            a.dimension.cmp(&b.dimension)
                .then(a.filtration_value.partial_cmp(&b.filtration_value).unwrap())
        });

        simplices
    }

    /// filtration → lophat 입력 형태로 변환
    pub fn to_lophat_columns(&self, filtration: &[FilteredSimplex]) -> Vec<VecColumn> {
        // simplex → 재인덱싱 (정렬 후 인덱스가 바뀜)
        let mut columns: Vec<VecColumn> = Vec::with_capacity(filtration.len());

        // 각 simplex를 lophat VecColumn으로 변환
        for simplex in filtration {
            let col: VecColumn = (simplex.dimension, simplex.boundary_indices.clone()).into();
            columns.push(col);
        }

        columns
    }
}
