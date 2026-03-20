class StructuralVerifier:
    def __init__(self, relation_index: dict, snippet_index: dict):
        self.relation_index = relation_index
        self.snippet_index = snippet_index

    def has_direct_relation(self, source_id: str, target_id: str) -> bool:
        return (source_id, target_id) in self.relation_index

    def evidence_overlap_score(self, source_id: str, target_id: str) -> float:
        s1 = self.snippet_index.get(source_id, "")
        s2 = self.snippet_index.get(target_id, "")
        if not s1 or not s2:
            return 0.0
        toks1 = set(s1.lower().split())
        toks2 = set(s2.lower().split())
        if not toks1 or not toks2:
            return 0.0
        return len(toks1 & toks2) / len(toks1 | toks2)

    def verify(self, hyp):
        # Claude Hypothesis: involved_entities / GPT Hypothesis: source_id, target_id
        source_id = getattr(hyp, "source_id", None) or (hyp.involved_entities[0] if hyp.involved_entities else "")
        target_id = getattr(hyp, "target_id", None) or (hyp.involved_entities[1] if len(hyp.involved_entities) > 1 else "")
        direct_exists = self.has_direct_relation(source_id, target_id)
        overlap = self.evidence_overlap_score(source_id, target_id)

        score = hyp.confidence
        if direct_exists:
            score -= 0.2
        if overlap > 0.1:
            score += 0.2

        return {
            "final_score": max(0.0, min(1.0, score)),
            "direct_relation_exists": direct_exists,
            "overlap_score": overlap
        }
