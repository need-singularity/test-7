from dataclasses import dataclass
from typing import Optional


@dataclass
class Entity:
    entity_id: str
    label: str
    description: Optional[str] = None


@dataclass
class Edge:
    source_id: str
    relation_id: str
    target_id: str
    weight: float = 1.0
    is_hierarchical: bool = False
