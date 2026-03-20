use nalgebra::DMatrix;

/// Compute Gromov δ-hyperbolicity of a distance matrix.
/// Uses the 4-point condition: for all x,y,z,w:
///   δ = max over all quadruples of:
///     (S₁+S₂ - max(S₁,S₂,S₃)) / 2
///   where S₁=d(x,y)+d(z,w), S₂=d(x,z)+d(y,w), S₃=d(x,w)+d(y,z)
/// O(n⁴) — only for small graphs (n ≤ 200)
pub fn delta_hyperbolicity(dm: &DMatrix<f64>) -> f64 {
    let n = dm.nrows();
    let mut delta = 0.0_f64;

    for x in 0..n {
        for y in (x+1)..n {
            for z in (y+1)..n {
                for w in (z+1)..n {
                    let s1 = dm[(x,y)] + dm[(z,w)];
                    let s2 = dm[(x,z)] + dm[(y,w)];
                    let s3 = dm[(x,w)] + dm[(y,z)];

                    let mut sums = [s1, s2, s3];
                    sums.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    // δ for this quadruple = (sums[0] - sums[1]) / 2
                    let d = (sums[0] - sums[1]) / 2.0;
                    delta = delta.max(d);
                }
            }
        }
    }

    delta
}

/// Hierarchy score: 1/(1+δ). Score of 1.0 = perfect tree, close to 0 = very non-tree-like
pub fn hierarchy_score(delta: f64) -> f64 {
    1.0 / (1.0 + delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_is_zero_hyperbolic() {
        // Path graph: 0-1-2-3, distances = hop count
        // Trees have δ=0
        let dm = DMatrix::from_row_slice(4, 4, &[
            0.0, 1.0, 2.0, 3.0,
            1.0, 0.0, 1.0, 2.0,
            2.0, 1.0, 0.0, 1.0,
            3.0, 2.0, 1.0, 0.0,
        ]);
        let delta = delta_hyperbolicity(&dm);
        assert!(delta < 0.001, "Tree should have δ≈0, got {}", delta);
        assert!((hierarchy_score(delta) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cycle_is_nonzero_hyperbolic() {
        // 4-cycle: 0-1-2-3-0, shortest path distances
        let dm = DMatrix::from_row_slice(4, 4, &[
            0.0, 1.0, 2.0, 1.0,
            1.0, 0.0, 1.0, 2.0,
            2.0, 1.0, 0.0, 1.0,
            1.0, 2.0, 1.0, 0.0,
        ]);
        let delta = delta_hyperbolicity(&dm);
        assert!(delta > 0.0, "Cycle should have δ>0, got {}", delta);
    }
}
