use super::persistence::PersistenceResult;

/// Betti 시그니처 — 위상학적 요약
#[derive(Debug, Clone)]
pub struct BettiSignature {
    /// beta_0: 연결 성분 수 (significant only)
    pub beta_0: usize,
    /// beta_1: 1차원 hole (순환/루프) 수
    pub beta_1: usize,
    /// beta_2: 2차원 void 수 (있다면)
    pub beta_2: usize,
    /// persistence entropy: -sum p_i log(p_i)
    pub persistence_entropy: f64,
    /// 총 persistence (모든 bar 길이 합)
    pub total_persistence: f64,
    /// 가장 긴 beta_1 bar들 (birth, death, persistence)
    pub long_bars_dim1: Vec<(f64, f64, f64)>,
    /// 평균 persistence (dim 1)
    pub mean_persistence_dim1: f64,
}

impl BettiSignature {
    /// PersistenceResult로부터 BettiSignature 추출
    pub fn from_persistence(result: &PersistenceResult, threshold: f64) -> Self {
        let significant_pairs: Vec<_> = result.pairs.iter()
            .filter(|p| p.persistence > threshold)
            .collect();

        let beta_0 = significant_pairs.iter()
            .filter(|p| p.dimension == 0)
            .count()
            + result.unpaired.iter()
                .filter(|u| u.dimension == 0)
                .count();

        let dim1_bars: Vec<_> = significant_pairs.iter()
            .filter(|p| p.dimension == 1)
            .collect();

        let beta_1 = dim1_bars.len();

        let beta_2 = significant_pairs.iter()
            .filter(|p| p.dimension == 2)
            .count();

        // Persistence entropy
        let all_persistences: Vec<f64> = result.pairs.iter()
            .filter(|p| p.persistence > 0.0)
            .map(|p| p.persistence)
            .collect();

        let total_persistence: f64 = all_persistences.iter().sum();
        let persistence_entropy = if total_persistence > 0.0 {
            all_persistences.iter()
                .map(|&p| {
                    let prob = p / total_persistence;
                    if prob > 0.0 { -prob * prob.ln() } else { 0.0 }
                })
                .sum()
        } else {
            0.0
        };

        // 가장 긴 dim1 bars
        let mut long_bars: Vec<(f64, f64, f64)> = dim1_bars.iter()
            .map(|p| (p.birth, p.death, p.persistence))
            .collect();
        long_bars.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let mean_persistence_dim1 = if !dim1_bars.is_empty() {
            dim1_bars.iter().map(|p| p.persistence).sum::<f64>() / dim1_bars.len() as f64
        } else {
            0.0
        };

        BettiSignature {
            beta_0,
            beta_1,
            beta_2,
            persistence_entropy,
            total_persistence,
            long_bars_dim1: long_bars,
            mean_persistence_dim1,
        }
    }
}
