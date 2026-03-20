from dataclasses import dataclass, field
from typing import List

@dataclass
class SolutionCandidate:
    mechanism: str
    hypothesis_ids: List[int] = field(default_factory=list)
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    topological_basis: str = ""
    problem_id: str = ""
