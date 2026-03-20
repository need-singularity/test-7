"""
EmergenceDetector: 6:3:1 weighted scoring
- topology (0.6): B1 count + max persistence
- hierarchy (0.3): delta-hyperbolicity -> 1/(1+delta)
- novelty (0.1): structural novelty (B1 rarity, topo-hier divergence)
"""
from dataclasses import dataclass

@dataclass
class EmergenceScore:
    topology_score: float    # 0.0-1.0
    hierarchy_score: float   # 0.0-1.0
    novelty_score: float     # 0.0-1.0
    total_score: float       # weighted combination
    interpretation: str      # "strong" / "moderate" / "weak"

    def to_dict(self) -> dict:
        return {
            "topology": self.topology_score,
            "hierarchy": self.hierarchy_score,
            "novelty": self.novelty_score,
            "total": self.total_score,
            "interpretation": self.interpretation,
        }

    def __getitem__(self, key):
        return self.to_dict()[key]

class EmergenceDetector:
    def __init__(
        self,
        topo_weight: float = 0.6,
        hier_weight: float = 0.3,
        novelty_weight: float = 0.1,
        beta1_saturation: int = 5,
        persistence_saturation: float = 2.0,
    ):
        self.topo_weight = topo_weight
        self.hier_weight = hier_weight
        self.novelty_weight = novelty_weight
        self.beta1_saturation = beta1_saturation
        self.persistence_saturation = persistence_saturation
        self._history = []  # past topology bundles for novelty comparison

    def score(self, topo_bundle: dict, hier_bundle: dict) -> EmergenceScore:
        """Compute 3-channel emergence score"""
        # Channel 1: Topology (B1 + persistence)
        beta1 = topo_bundle.get("beta1", 0)
        max_pers = topo_bundle.get("max_persistence_h1", 0.0)

        cycle_score = min(1.0, beta1 / self.beta1_saturation)
        pers_score = min(1.0, max_pers / self.persistence_saturation)
        topology_score = 0.6 * cycle_score + 0.4 * pers_score

        # Channel 2: Hierarchy (already normalized)
        hierarchy_score = hier_bundle.get("hierarchy_score", 0.5)

        # Channel 3: Novelty (structural)
        novelty_score = self._compute_novelty(topo_bundle, hierarchy_score)

        # Weighted combination
        total = (
            self.topo_weight * topology_score
            + self.hier_weight * hierarchy_score
            + self.novelty_weight * novelty_score
        )

        # Interpretation
        if total >= 0.70:
            interpretation = "strong"
        elif total >= 0.45:
            interpretation = "moderate"
        else:
            interpretation = "weak"

        # Record for novelty history
        self._history.append({
            "beta1": beta1,
            "max_persistence": max_pers,
            "hierarchy": hierarchy_score,
        })

        return EmergenceScore(
            topology_score=round(topology_score, 4),
            hierarchy_score=round(hierarchy_score, 4),
            novelty_score=round(novelty_score, 4),
            total_score=round(total, 4),
            interpretation=interpretation,
        )

    def _compute_novelty(self, topo_bundle: dict, hierarchy_score: float) -> float:
        """
        Structural novelty:
        - B1 rarity: how unusual is this B1 count vs history
        - topo-hier divergence: B1 high AND hierarchy high = interesting mix
        - cold-start: 0.5 default
        """
        if len(self._history) < 2:
            return 0.5  # cold-start

        beta1 = topo_bundle.get("beta1", 0)

        # B1 rarity: compare to historical mean
        hist_betas = [h["beta1"] for h in self._history]
        mean_beta = sum(hist_betas) / len(hist_betas)
        rarity = min(1.0, abs(beta1 - mean_beta) / (mean_beta + 1))

        # topo-hier divergence: both high = emergence candidate
        topo_norm = min(1.0, beta1 / self.beta1_saturation)
        divergence = topo_norm * hierarchy_score  # both must be high

        return 0.5 * rarity + 0.5 * divergence
