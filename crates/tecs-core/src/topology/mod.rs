pub mod vr_complex;
pub mod persistence;
pub mod betti;
pub mod emergence;

pub use persistence::{PersistenceComputer, PersistenceResult, PersistencePair};
pub use betti::BettiSignature;
pub use emergence::{EmergenceDetector, EmergenceSignal, EpsilonAction, EmergenceDetectorConfig, ThresholdMode};
