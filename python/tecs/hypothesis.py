"""
Flow 5: 규칙 기반 가설 생성기

위상학적 구조(beta_1 hole, gap, bridge)에서 missing link / mediator concept 제안
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class Hypothesis:
    """생성된 가설"""
    hypothesis_type: str         # "missing_link" | "mediator_concept" | "latent_hub" | "bridge_edge"
    description: str             # 사람이 읽을 수 있는 설명
    involved_entities: List[str] # 관련 Wikidata ID들
    involved_labels: List[str]   # 사람이 읽을 수 있는 레이블
    confidence: float            # 0.0 ~ 1.0
    evidence_needed: str         # 검증에 필요한 증거 유형
    topological_basis: str       # 위상학적 근거
    birth_scale: float = 0.0    # 이 구조가 나타나는 거리 스케일
    death_scale: float = 0.0    # 사라지는 스케일
    persistence: float = 0.0    # 안정성


@dataclass
class ReasoningContext:
    """추론 컨텍스트"""
    query_entities: List[str]
    subgraph_nodes: List[str]
    subgraph_labels: Dict[str, str]
    subgraph_edges: List[Tuple[str, str, str]]  # (src, dst, relation)
    beta_0: int
    beta_1: int
    long_bars: List[Tuple[float, float, float]]  # (birth, death, persistence)
    epsilon: float
    iteration: int


class HypothesisGenerator:
    """규칙 기반 가설 생성기"""

    def __init__(self):
        self.rules = [
            self._rule_persistent_hole,
            self._rule_disconnected_clusters,
            self._rule_near_cycle,
            self._rule_hub_convergence,
        ]

    def generate(self, ctx: ReasoningContext) -> List[Hypothesis]:
        """모든 규칙을 적용하여 가설 목록 생성"""
        hypotheses = []
        for rule in self.rules:
            hyps = rule(ctx)
            hypotheses.extend(hyps)

        # confidence 순으로 정렬
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses

    def _rule_persistent_hole(self, ctx: ReasoningContext) -> List[Hypothesis]:
        """
        규칙 1: Persistent beta_1 hole -> Missing Link

        beta_1 bar가 길게 살아남으면, 그 cycle을 구성하는 노드들 사이에
        직접 연결이 없지만 의미적으로 강한 관계가 있을 수 있음
        """
        hypotheses = []

        for (birth, death, persistence) in ctx.long_bars:
            if persistence < 0.1:
                continue

            # 높은 persistence = 강한 구조적 증거
            confidence = min(0.9, persistence / (death + 1e-10))

            hyp = Hypothesis(
                hypothesis_type="missing_link",
                description=(
                    f"Persistent topological hole detected (beta_1) with lifetime "
                    f"[{birth:.3f}, {death:.3f}] (persistence={persistence:.3f}). "
                    f"This suggests a structurally significant gap between concepts "
                    f"that may have an undocumented relationship."
                ),
                involved_entities=ctx.query_entities,
                involved_labels=[ctx.subgraph_labels.get(e, e) for e in ctx.query_entities],
                confidence=confidence,
                evidence_needed="cross_reference_text",
                topological_basis=f"beta_1 bar: birth={birth:.3f}, death={death:.3f}",
                birth_scale=birth,
                death_scale=death,
                persistence=persistence,
            )
            hypotheses.append(hyp)

        return hypotheses

    def _rule_disconnected_clusters(self, ctx: ReasoningContext) -> List[Hypothesis]:
        """
        규칙 2: beta_0 > 1 -> 분리된 개념 클러스터 -> Bridge Edge 제안

        쌍곡 공간에서 가까운데 그래프에서 연결이 없는 경우
        """
        if ctx.beta_0 <= 1:
            return []

        confidence = min(0.8, 0.3 + ctx.beta_0 * 0.1)

        return [Hypothesis(
            hypothesis_type="bridge_edge",
            description=(
                f"Found {ctx.beta_0} disconnected clusters in the subgraph. "
                f"These clusters are geometrically close in hyperbolic space "
                f"but lack explicit graph connections, suggesting potential bridges."
            ),
            involved_entities=ctx.query_entities,
            involved_labels=[ctx.subgraph_labels.get(e, e) for e in ctx.query_entities],
            confidence=confidence,
            evidence_needed="category_overlap_check",
            topological_basis=f"beta_0={ctx.beta_0} (>1 indicates disconnection)",
        )]

    def _rule_near_cycle(self, ctx: ReasoningContext) -> List[Hypothesis]:
        """
        규칙 3: 짧은 persistence beta_1 -> 거의 닫히는 cycle -> Mediator Concept

        cycle이 거의 완성되지만 한 edge가 부족한 경우,
        그 gap을 메울 중간 개념 제안
        """
        hypotheses = []

        short_bars = [(b, d, p) for (b, d, p) in ctx.long_bars if 0.01 < p < 0.1]

        for (birth, death, persistence) in short_bars:
            confidence = 0.4 + (0.1 - persistence) * 3  # 짧을수록 더 확신

            hyp = Hypothesis(
                hypothesis_type="mediator_concept",
                description=(
                    f"Near-closing cycle detected: a topological loop almost closes "
                    f"but has a small gap (persistence={persistence:.3f}). "
                    f"A mediating concept may bridge this gap."
                ),
                involved_entities=ctx.query_entities,
                involved_labels=[ctx.subgraph_labels.get(e, e) for e in ctx.query_entities],
                confidence=min(0.85, confidence),
                evidence_needed="mediator_search",
                topological_basis=f"Near-cycle: persistence={persistence:.3f} (small but nonzero)",
                birth_scale=birth,
                death_scale=death,
                persistence=persistence,
            )
            hypotheses.append(hyp)

        return hypotheses

    def _rule_hub_convergence(self, ctx: ReasoningContext) -> List[Hypothesis]:
        """
        규칙 4: 여러 경로가 특정 노드로 수렴 -> Latent Hub Concept

        서브그래프에서 degree가 비정상적으로 높은 노드
        """
        # edge 목록에서 degree 계산
        degree_count: Dict[str, int] = {}
        for (src, dst, _rel) in ctx.subgraph_edges:
            degree_count[src] = degree_count.get(src, 0) + 1
            degree_count[dst] = degree_count.get(dst, 0) + 1

        if not degree_count:
            return []

        avg_degree = sum(degree_count.values()) / len(degree_count)
        hypotheses = []

        for node_id, deg in degree_count.items():
            if deg > avg_degree * 2 and node_id not in ctx.query_entities:
                label = ctx.subgraph_labels.get(node_id, node_id)
                confidence = min(0.75, 0.3 + (deg / (avg_degree + 1)) * 0.1)
                hyp = Hypothesis(
                    hypothesis_type="latent_hub",
                    description=(
                        f"Node '{label}' ({node_id}) has unusually high degree ({deg}) "
                        f"compared to average ({avg_degree:.1f}), suggesting it may be "
                        f"a latent hub concept connecting multiple knowledge domains."
                    ),
                    involved_entities=[node_id] + ctx.query_entities,
                    involved_labels=[label] + [ctx.subgraph_labels.get(e, e) for e in ctx.query_entities],
                    confidence=confidence,
                    evidence_needed="hub_analysis",
                    topological_basis=f"degree={deg}, avg={avg_degree:.1f}",
                )
                hypotheses.append(hyp)

        return hypotheses


class HypothesisEngine:
    """
    Dict 기반 인터페이스 (orchestrator/PyO3 연동용)

    graph_bundle, topology_bundle dict를 받아 가설 생성.
    내부적으로 HypothesisGenerator의 4규칙을 사용.
    """

    def __init__(self):
        self._generator = HypothesisGenerator()

    def generate(self, graph_bundle: dict, topology_bundle: dict) -> list[Hypothesis]:
        """dict 기반 입력 -> ReasoningContext 변환 -> 4규칙 실행"""
        ctx = ReasoningContext(
            query_entities=graph_bundle.get("seed_entities", []),
            subgraph_nodes=graph_bundle.get("subgraph_nodes", []),
            subgraph_labels=graph_bundle.get("subgraph_labels", {}),
            subgraph_edges=graph_bundle.get("subgraph_edges", []),
            beta_0=topology_bundle.get("beta0", 1),
            beta_1=topology_bundle.get("beta1", 0),
            long_bars=[
                (b, d, d - b) for b, d in topology_bundle.get("long_h1", [])
            ],
            epsilon=graph_bundle.get("epsilon", 2.0),
            iteration=graph_bundle.get("iteration", 0),
        )
        return self._generator.generate(ctx)
