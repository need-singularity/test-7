from tecs.entity_linker import EntityLinker
from tecs.hypothesis import HypothesisEngine
from tecs.verifier import StructuralVerifier
from tecs.explainer import Explainer

class TECSOrchestrator:
    def __init__(self, rust_engine, alias_index, relation_index, snippet_index):
        self.rust_engine = rust_engine
        self.linker = EntityLinker(alias_index)
        self.h_engine = HypothesisEngine()
        self.verifier = StructuralVerifier(relation_index, snippet_index)
        self.explainer = Explainer()

    def run(self, query: str) -> dict:
        linked = self.linker.link(query)
        if not linked:
            return {"status": "no_entity"}

        seed_ids = [x.entity_id for x in linked[:2]]

        # Rust 파이프라인
        graph_bundle = self.rust_engine.build_candidate_graph(seed_ids, hops=2, epsilon=2.0, max_nodes=300)

        # Use edges-based topology if available, otherwise fallback to stub
        if hasattr(self.rust_engine, 'compute_topology_from_edges'):
            nodes = graph_bundle.get("subgraph_nodes", seed_ids)
            node_map = {n: i for i, n in enumerate(nodes)}
            edges = []
            for src, dst, _rel in graph_bundle.get("subgraph_edges", []):
                if src in node_map and dst in node_map:
                    edges.append((node_map[src], node_map[dst]))
            if edges:
                topo_bundle = self.rust_engine.compute_topology_from_edges(edges, len(node_map))
            else:
                topo_bundle = self.rust_engine.compute_topology(graph_bundle)
        else:
            topo_bundle = self.rust_engine.compute_topology(graph_bundle)

        # Python reasoning
        hyps = self.h_engine.generate(graph_bundle, topo_bundle)

        verified = []
        for hyp in hyps:
            v = self.verifier.verify(hyp)
            explanation = self.explainer.explain(hyp, v)
            verified.append({
                "hypothesis": hyp,
                "verification": v,
                "explanation": explanation
            })

        return {
            "status": "ok",
            "linked": linked,
            "graph_bundle": graph_bundle,
            "topology_bundle": topo_bundle,
            "results": verified,
        }

    def solve(self, problem) -> dict:
        """Problem -> entity linking -> subgraph -> topology -> hypothesis -> SolutionCandidate"""
        from tecs.solution import SolutionCandidate
        from tecs.emergence import EmergenceDetector

        # Fix: resolve each entity independently, not " ".join()
        linked = self.linker.resolve_entities(problem.domain_entities)
        if not linked:
            return {"status": "no_entity", "solutions": []}

        seed_ids = [x.entity_id for x in linked]

        # Build candidate graph
        graph_bundle = self.rust_engine.build_candidate_graph(seed_ids, hops=2, epsilon=2.0, max_nodes=300)

        # Topology (edges-based if available)
        nodes = graph_bundle.get("subgraph_nodes", seed_ids)
        node_map = {n: i for i, n in enumerate(nodes)}
        edges = []
        for src, dst, _rel in graph_bundle.get("subgraph_edges", []):
            if src in node_map and dst in node_map:
                edges.append((node_map[src], node_map[dst]))

        if edges and hasattr(self.rust_engine, 'compute_topology_from_edges'):
            topo_bundle = self.rust_engine.compute_topology_from_edges(edges, len(node_map))
        else:
            topo_bundle = self.rust_engine.compute_topology(graph_bundle)

        # Hierarchy
        hier_bundle = {"hierarchy_score": 0.5}
        if edges and hasattr(self.rust_engine, 'compute_hyperbolicity'):
            hier_bundle = self.rust_engine.compute_hyperbolicity(edges, len(node_map))

        # Emergence
        detector = EmergenceDetector()
        emergence = detector.score(topo_bundle, hier_bundle)

        # Hypotheses
        hyps = self.h_engine.generate(graph_bundle, topo_bundle)

        # Verify and build solutions
        solutions = []
        for i, hyp in enumerate(hyps):
            v = self.verifier.verify(hyp)
            explanation = self.explainer.explain(hyp, v)
            solutions.append(SolutionCandidate(
                mechanism=getattr(hyp, "description", "") or getattr(hyp, "rationale", ""),
                hypothesis_ids=[i],
                novelty_score=v["final_score"],
                feasibility_score=1.0 - v.get("overlap_score", 0.0),
                topological_basis=getattr(hyp, "topological_basis", ""),
                problem_id=problem.problem_id,
            ))

        return {
            "status": "ok",
            "problem": problem,
            "linked": linked,
            "topology_bundle": topo_bundle,
            "hierarchy_bundle": hier_bundle,
            "emergence_score": emergence,
            "solutions": solutions,
        }
