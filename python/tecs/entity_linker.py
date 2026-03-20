import re
from dataclasses import dataclass
from typing import List

QID_PATTERN = re.compile(r'^Q\d+$', re.IGNORECASE)


@dataclass
class LinkedEntity:
    entity_id: str
    label: str
    score: float


class EntityLinker:
    def __init__(self, alias_index: dict[str, list[dict]]):
        self.alias_index = alias_index

    def normalize(self, text: str) -> str:
        return text.strip().lower()

    def link(self, query: str, top_k: int = 5) -> List[LinkedEntity]:
        tokens = [self.normalize(t) for t in query.replace(",", " ").split()]
        candidates = []
        for tok in tokens:
            for item in self.alias_index.get(tok, []):
                candidates.append(
                    LinkedEntity(
                        entity_id=item["entity_id"],
                        label=item["label"],
                        score=item.get("score", 0.5),
                    )
                )
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def resolve_entity(self, entity_id: str) -> 'LinkedEntity':
        """Resolve a single entity — QID passthrough or alias lookup"""
        if QID_PATTERN.match(entity_id):
            return LinkedEntity(entity_id=entity_id.upper(), label=entity_id.upper(), score=1.0)
        results = self.link(entity_id, top_k=1)
        return results[0] if results else None

    def resolve_entities(self, entities: list[str]) -> list['LinkedEntity']:
        """Resolve multiple entities independently"""
        results = []
        for e in entities:
            resolved = self.resolve_entity(e)
            if resolved:
                results.append(resolved)
        return results
