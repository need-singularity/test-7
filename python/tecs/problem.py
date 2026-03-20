from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import json

@dataclass
class Problem:
    title: str
    domain_entities: List[str]  # Wikidata QIDs
    constraints: List[str] = field(default_factory=list)  # free text
    goals: List[str] = field(default_factory=list)

    @property
    def problem_id(self) -> str:
        content = json.dumps({
            "title": self.title,
            "domain_entities": sorted(self.domain_entities),
            "constraints": self.constraints,
            "goals": self.goals,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
