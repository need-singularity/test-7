class Explainer:
    def explain(self, hyp, verify_bundle: dict) -> str:
        msg = []
        # Claude Hypothesis: hypothesis_type, involved_entities
        # GPT Hypothesis: kind, source_id, target_id
        kind = getattr(hyp, "kind", None) or getattr(hyp, "hypothesis_type", "unknown")
        entities = getattr(hyp, "involved_entities", [])
        source = getattr(hyp, "source_id", None) or (entities[0] if entities else "?")
        target = getattr(hyp, "target_id", None) or (entities[1] if len(entities) > 1 else "?")

        msg.append(f"[{kind}] {source} -> {target}")
        mediators = getattr(hyp, "mediator_ids", [])
        if mediators:
            msg.append(f"mediators={', '.join(mediators)}")
        rationale = getattr(hyp, "rationale", None) or getattr(hyp, "description", "")
        msg.append(f"rationale={rationale}")
        msg.append(f"verified_score={verify_bundle['final_score']:.2f}")
        if verify_bundle["direct_relation_exists"]:
            msg.append("note=direct relation already exists; treat as redundancy check")
        else:
            msg.append("note=no direct relation found; candidate missing-link")
        return " | ".join(msg)
