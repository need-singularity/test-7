"""Smoke tests for Problem → SolutionCandidate pipeline"""
import sys
sys.path.insert(0, "python")

from tecs.problem import Problem
from tecs.orchestrator import TECSOrchestrator

class MockEngine:
    def build_candidate_graph(self, seed_ids, hops=2, epsilon=2.0, max_nodes=300):
        return {
            "seed_entities": seed_ids,
            "subgraph_nodes": seed_ids,
            "subgraph_labels": {e: e for e in seed_ids},
            "subgraph_edges": [],
            "candidate_cycle_nodes": seed_ids[:4],
            "bridge_candidates": seed_ids[:2],
        }
    def compute_topology(self, bundle):
        return {"beta0": 2, "beta1": 1, "long_h1": [(0.2, 0.8)]}

def make_orchestrator():
    return TECSOrchestrator(
        rust_engine=MockEngine(),
        alias_index={
            "quantum": [{"entity_id": "Q11426", "label": "Quantum mechanics", "score": 0.9}],
            "gravity": [{"entity_id": "Q11412", "label": "General relativity", "score": 0.9}],
            "p": [{"entity_id": "Q178692", "label": "P vs NP", "score": 0.8}],
            "np": [{"entity_id": "Q178692", "label": "P vs NP", "score": 0.8}],
            "q11426": [{"entity_id": "Q11426", "label": "Quantum mechanics", "score": 0.9}],
            "q11412": [{"entity_id": "Q11412", "label": "General relativity", "score": 0.9}],
            "q178692": [{"entity_id": "Q178692", "label": "P vs NP", "score": 0.8}],
        },
        relation_index={},
        snippet_index={
            "Q11426": "Quantum mechanics describes nature at atomic scale.",
            "Q11412": "General relativity describes gravity as spacetime curvature.",
            "Q178692": "P versus NP is a major unsolved problem in computer science.",
        },
    )

def test_quantum_gravity():
    app = make_orchestrator()
    p = Problem(title="Quantum Gravity", domain_entities=["Q11426", "Q11412"], goals=["unification"])
    result = app.solve(p)
    assert result["status"] == "ok"
    assert len(result["solutions"]) > 0
    for s in result["solutions"]:
        assert s.problem_id == p.problem_id
        assert 0.0 <= s.novelty_score <= 1.0
    print(f"  quantum_gravity: {len(result['solutions'])} solutions")

def test_p_vs_np():
    app = make_orchestrator()
    p = Problem(title="P vs NP", domain_entities=["Q178692"], goals=["proof or disproof"])
    result = app.solve(p)
    assert result["status"] == "ok"
    print(f"  p_vs_np: {len(result['solutions'])} solutions")

def test_empty_problem():
    app = make_orchestrator()
    p = Problem(title="Empty", domain_entities=[], goals=[])
    result = app.solve(p)
    # Should not crash, may return no_entity
    print(f"  empty: status={result['status']}")

if __name__ == "__main__":
    test_quantum_gravity()
    test_p_vs_np()
    test_empty_problem()
    print("\nAll problem solver tests passed!")
