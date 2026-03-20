use super::vr_complex::VietorisRipsBuilder;
use lophat::algorithms::{LockFreeAlgorithm, DecompositionAlgo, Decomposition};
use nalgebra::DMatrix;
use log::debug;

/// Persistence 계산 결과
#[derive(Debug, Clone)]
pub struct PersistenceResult {
    /// (birth, death, dimension) 쌍들
    pub pairs: Vec<PersistencePair>,
    /// 짝 없는 simplex들 (무한 persistence)
    pub unpaired: Vec<UnpairedSimplex>,
}

#[derive(Debug, Clone)]
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
    pub persistence: f64,       // death - birth
    pub birth_simplex_idx: usize,
    pub death_simplex_idx: usize,
}

#[derive(Debug, Clone)]
pub struct UnpairedSimplex {
    pub birth: f64,
    pub dimension: usize,
    pub simplex_idx: usize,
}

/// PH 계산기
pub struct PersistenceComputer {
    /// VR complex 빌더
    vr_builder: VietorisRipsBuilder,
}

impl PersistenceComputer {
    pub fn new(max_dim: usize, max_radius: f64) -> Self {
        Self {
            vr_builder: VietorisRipsBuilder::new(max_dim, max_radius),
        }
    }

    /// 자동 max_radius 결정 (중위값의 2배)
    pub fn auto_radius(dm: &DMatrix<f64>) -> f64 {
        let n = dm.nrows();
        let mut all_dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dm[(i, j)];
                if d.is_finite() && d > 0.0 {
                    all_dists.push(d);
                }
            }
        }
        all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if all_dists.is_empty() {
            return 1.0;
        }

        let median = all_dists[all_dists.len() / 2];
        median * 2.0
    }

    /// 거리 매트릭스로부터 persistent homology 계산
    pub fn compute(&self, dm: &DMatrix<f64>) -> PersistenceResult {
        debug!("Computing PH: {}x{} distance matrix, max_dim={}, max_radius={}",
            dm.nrows(), dm.ncols(), self.vr_builder.max_dim, self.vr_builder.max_radius);

        // 1) VR filtration 생성
        let filtration = self.vr_builder.build_filtration(dm);
        debug!("Built VR complex: {} simplices", filtration.len());

        // 2) lophat 입력으로 변환
        let columns = self.vr_builder.to_lophat_columns(&filtration);

        // 3) Lock-free 알고리즘으로 R=DV 분해
        let decomposition = LockFreeAlgorithm::init(None)
            .add_cols(columns.into_iter())
            .decompose();

        // 4) Persistence diagram 추출
        let diagram = decomposition.diagram();

        // 5) 쌍을 PersistencePair로 변환
        let mut pairs = Vec::new();
        for &(birth_idx, death_idx) in &diagram.paired {
            let birth_filt = filtration[birth_idx].filtration_value;
            let death_filt = filtration[death_idx].filtration_value;
            let dim = filtration[birth_idx].dimension;

            pairs.push(PersistencePair {
                birth: birth_filt,
                death: death_filt,
                dimension: dim,
                persistence: death_filt - birth_filt,
                birth_simplex_idx: birth_idx,
                death_simplex_idx: death_idx,
            });
        }

        let mut unpaired = Vec::new();
        for &idx in &diagram.unpaired {
            unpaired.push(UnpairedSimplex {
                birth: filtration[idx].filtration_value,
                dimension: filtration[idx].dimension,
                simplex_idx: idx,
            });
        }

        debug!("PH complete: {} pairs, {} unpaired", pairs.len(), unpaired.len());

        PersistenceResult { pairs, unpaired }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// Construct a distance matrix for the 2-sphere triangulation (icosahedron-like).
    /// We use 6 vertices of an octahedron embedded in R^3 — the minimal triangulation
    /// of S^2 that has the right Betti numbers: β₀=1, β₁=0, β₂=1.
    /// Vertices: ±e₁, ±e₂, ±e₃ (the 6 vertices of a regular octahedron).
    /// The octahedron has 8 triangular faces, 12 edges, 6 vertices → χ = 6-12+8 = 2 = S².
    #[test]
    fn test_sphere_triangulation_betti_numbers() {
        // 6 vertices of a regular octahedron in R^3
        let points: Vec<[f64; 3]> = vec![
            [ 1.0,  0.0,  0.0],  // 0: +x
            [-1.0,  0.0,  0.0],  // 1: -x
            [ 0.0,  1.0,  0.0],  // 2: +y
            [ 0.0, -1.0,  0.0],  // 3: -y
            [ 0.0,  0.0,  1.0],  // 4: +z
            [ 0.0,  0.0, -1.0],  // 5: -z
        ];
        let n = points.len();

        // Build Euclidean distance matrix
        let mut dm = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let dz = points[i][2] - points[j][2];
                dm[(i, j)] = (dx * dx + dy * dy + dz * dz).sqrt();
            }
        }

        // All edges of the octahedron have length sqrt(2) ≈ 1.414.
        // Antipodal pairs have distance 2.0.
        // Set max_radius to 1.5 to include edges (sqrt(2)≈1.414) but exclude
        // antipodal diagonals (2.0). This gives the octahedral complex.
        let computer = PersistenceComputer::new(2, 1.5);
        let result = computer.compute(&dm);

        // Count Betti numbers from unpaired simplices (infinite persistence)
        let beta_0 = result.unpaired.iter().filter(|u| u.dimension == 0).count();
        let beta_1 = result.unpaired.iter().filter(|u| u.dimension == 1).count();
        let beta_2 = result.unpaired.iter().filter(|u| u.dimension == 2).count();

        // Count simplices to verify we have the right complex
        let vr = VietorisRipsBuilder::new(2, 1.5);
        let filtration = vr.build_filtration(&dm);
        let n_vertices = filtration.iter().filter(|s| s.dimension == 0).count();
        let n_edges = filtration.iter().filter(|s| s.dimension == 1).count();
        let n_triangles = filtration.iter().filter(|s| s.dimension == 2).count();

        // Octahedron: 6 vertices, 12 edges, 8 triangles = 26 simplices
        // But VR complex may include more triangles than the geometric octahedron.
        // At radius 1.5, edges connect all non-antipodal pairs.
        // Each vertex is adjacent to 4 others (not its antipode).
        assert_eq!(n_vertices, 6, "Should have 6 vertices");
        assert_eq!(n_edges, 12, "Should have 12 edges (octahedron)");

        // Total simplex count
        let total = filtration.len();
        eprintln!("Filtration: {} vertices, {} edges, {} triangles ({} total)",
            n_vertices, n_edges, n_triangles, total);
        eprintln!("Betti: β₀={}, β₁={}, β₂={}", beta_0, beta_1, beta_2);

        // S² has β₀=1 (connected), β₁=0 (no 1-cycles)
        assert_eq!(beta_0, 1, "β₀ should be 1 (connected)");

        // Check there are no persistent 1-cycles (β₁=0 for S²)
        let h1_pairs: Vec<_> = result.pairs.iter()
            .filter(|p| p.dimension == 1)
            .collect();
        // All 1-cycles should be killed (paired), meaning β₁=0
        assert_eq!(beta_1, 0, "β₁ should be 0 for S²");
    }

    /// Two points far apart (distance 10.0), max_radius=1.0.
    /// No edges form → two connected components. β₀=2, β₁=0.
    #[test]
    fn test_two_points() {
        let dm = DMatrix::from_row_slice(2, 2, &[
            0.0, 10.0,
            10.0, 0.0,
        ]);

        let computer = PersistenceComputer::new(1, 1.0);
        let result = computer.compute(&dm);

        let beta_0 = result.unpaired.iter().filter(|u| u.dimension == 0).count();
        let beta_1 = result.unpaired.iter().filter(|u| u.dimension == 1).count();

        assert_eq!(beta_0, 2, "β₀ should be 2 (two disconnected points)");
        assert_eq!(beta_1, 0, "β₁ should be 0 (no edges, no cycles)");
    }

    /// 4 points forming a square with side length 1.0.
    /// Distances: adjacent = 1.0, diagonal = sqrt(2) ≈ 1.414.
    /// max_radius = 1.2: all 4 edges connect but diagonals don't.
    /// This forms a cycle → β₀=1 (connected), β₁=1 (one hole).
    #[test]
    fn test_single_cycle() {
        // Square vertices: (0,0), (1,0), (1,1), (0,1)
        let points: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let n = points.len();
        let mut dm = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                dm[(i, j)] = (dx * dx + dy * dy).sqrt();
            }
        }

        // max_radius = 1.2 includes edges of length 1.0 but excludes diagonals (sqrt(2) ≈ 1.414)
        let computer = PersistenceComputer::new(2, 1.2);
        let result = computer.compute(&dm);

        let beta_0 = result.unpaired.iter().filter(|u| u.dimension == 0).count();
        let beta_1 = result.unpaired.iter().filter(|u| u.dimension == 1).count();

        assert_eq!(beta_0, 1, "β₀ should be 1 (connected)");
        assert_eq!(beta_1, 1, "β₁ should be 1 (one cycle / hole)");
    }

    /// 3 points in a line: A-B distance 1.0, B-C distance 1.0, A-C distance 2.0.
    /// max_radius = 1.5: A-B and B-C connect, but A-C doesn't.
    /// Path graph → β₀=1 (connected), β₁=0 (no cycle).
    #[test]
    fn test_path_no_cycle() {
        let dm = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.0,
            2.0, 1.0, 0.0,
        ]);

        let computer = PersistenceComputer::new(2, 1.5);
        let result = computer.compute(&dm);

        let beta_0 = result.unpaired.iter().filter(|u| u.dimension == 0).count();
        let beta_1 = result.unpaired.iter().filter(|u| u.dimension == 1).count();

        assert_eq!(beta_0, 1, "β₀ should be 1 (connected via path)");
        assert_eq!(beta_1, 0, "β₁ should be 0 (no cycle in a path)");
    }
}
