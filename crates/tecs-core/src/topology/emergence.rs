use super::betti::BettiSignature;

/// 창발 신호 — 위상 변화 감지
#[derive(Debug, Clone)]
pub struct EmergenceSignal {
    /// beta_1 변화량
    pub delta_beta1: i64,
    /// persistence entropy 변화
    pub delta_entropy: f64,
    /// 복잡도 지표 omega = beta_0 + beta_1 + beta_2
    pub omega: f64,
    /// 창발 감지 여부
    pub is_emergence: bool,
    /// 혼돈 감지 여부 (omega > threshold)
    pub is_chaos: bool,
    /// 정체 감지 여부
    pub is_stagnant: bool,
    /// 권장 epsilon 조정 방향
    pub epsilon_action: EpsilonAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpsilonAction {
    /// epsilon를 확장 (탐색 범위 넓힘)
    Expand(f64),
    /// epsilon를 축소 (혼돈 억제)
    Shrink(f64),
    /// epsilon 유지
    Hold,
    /// 수렴 완료
    Converged,
}

/// 창발 감지기
pub struct EmergenceDetector {
    /// omega 최대값 (이상이면 chaos)
    pub omega_max: f64,
    /// epsilon 확장 비율
    pub expand_ratio: f64,
    /// epsilon 축소 비율
    pub shrink_ratio: f64,
    /// 정체 판단 entropy 임계치
    pub stagnant_threshold: f64,
    /// 정체 카운터 (연속 정체 시 수렴 판단)
    pub stagnant_patience: usize,
    /// 현재 정체 카운터
    stagnant_count: usize,
}

impl EmergenceDetector {
    pub fn new() -> Self {
        Self {
            omega_max: 50.0,
            expand_ratio: 1.3,
            shrink_ratio: 0.7,
            stagnant_threshold: 0.01,
            stagnant_patience: 3,
            stagnant_count: 0,
        }
    }

    /// 두 시그니처 비교 -> 창발 신호 생성
    pub fn detect(
        &mut self,
        prev: &BettiSignature,
        curr: &BettiSignature,
        current_epsilon: f64,
    ) -> EmergenceSignal {
        let delta_beta1 = curr.beta_1 as i64 - prev.beta_1 as i64;
        let delta_entropy = curr.persistence_entropy - prev.persistence_entropy;
        let omega = (curr.beta_0 + curr.beta_1 + curr.beta_2) as f64;

        let is_chaos = omega > self.omega_max;
        let is_emergence = delta_beta1 > 0
            && curr.total_persistence > prev.total_persistence;
        let is_stagnant = delta_beta1 == 0
            && delta_entropy.abs() < self.stagnant_threshold;

        // epsilon 조정 결정
        let epsilon_action = if is_emergence {
            self.stagnant_count = 0;
            EpsilonAction::Expand(current_epsilon * self.expand_ratio)
        } else if is_chaos {
            self.stagnant_count = 0;
            EpsilonAction::Shrink(current_epsilon * self.shrink_ratio)
        } else if is_stagnant {
            self.stagnant_count += 1;
            if self.stagnant_count >= self.stagnant_patience {
                EpsilonAction::Converged
            } else {
                EpsilonAction::Hold
            }
        } else {
            self.stagnant_count = 0;
            EpsilonAction::Hold
        };

        EmergenceSignal {
            delta_beta1,
            delta_entropy,
            omega,
            is_emergence,
            is_chaos,
            is_stagnant,
            epsilon_action,
        }
    }

    /// 정체 카운터 리셋
    pub fn reset(&mut self) {
        self.stagnant_count = 0;
    }
}

/// GHS-TDA 6:3:1 weighted configuration for emergence detection
#[derive(Debug, Clone)]
pub struct EmergenceDetectorConfig {
    pub hyperbolic_weight: f64,    // α = 0.6
    pub structural_weight: f64,    // β = 0.3
    pub confidence_weight: f64,    // γ = 0.1
    pub omega_min: f64,            // 0.1
    pub omega_max: f64,            // 0.5
    pub epsilon_growth: f64,       // 1.3
    pub epsilon_shrink: f64,       // 0.7
    pub threshold_mode: ThresholdMode,
}

#[derive(Debug, Clone)]
pub enum ThresholdMode {
    Fixed,
    Percentile { lower: f64, upper: f64 },
}

impl Default for EmergenceDetectorConfig {
    fn default() -> Self {
        Self {
            hyperbolic_weight: 0.6,
            structural_weight: 0.3,
            confidence_weight: 0.1,
            omega_min: 0.1,
            omega_max: 0.5,
            epsilon_growth: 1.3,
            epsilon_shrink: 0.7,
            threshold_mode: ThresholdMode::Fixed,
        }
    }
}
