# Experiments A-D: Topological Wall Passage Extensions

## Overview

Phase 1-5에서 확립한 LLM latent space의 위상적 벽(β₁ hole) 감지-통과 파이프라인을 4가지 방향으로 확장한다.

- **Exp-A**: Ricci flow 기반 벽 통과 (방향 보존)
- **Exp-B**: 쌍곡 거리 기반 PH (유클리드 vs 쌍곡 β₁ 비교)
- **Exp-C**: 쌍곡 Ricci flow (A+B 결합)
- **Exp-D**: Novel idea 생성 검증 (인간 평가 포함)

## Dependencies

```
Exp-D (독립)
Exp-A (독립)
Exp-B (독립)
Exp-A + Exp-B → Exp-C (둘 다 필요)
```

## Common Infrastructure

### `experiments/common.py`

기존 phase 파일들에서 반복되는 함수를 추출:

- `load_model()` — Llama 3.1 8B GGUF 로드
- `extract_embeddings(llm, prompt)` — prefix 누적 + suffix 변형 → point cloud
- `compute_full_topology(points)` — PCA → Ripser PH → walls/β₁/persistence
- `radial_perturbation(embeddings, wall, alpha)` — 단일 wall radial 수축
- `multi_wall_perturbation(embeddings, walls, alpha, max_walls)` — 다중 wall 수축
- `emergence_score(topo_before, topo_after)` — 3-channel passage score
- `get_passage_direction(embeddings, wall)` — cocycle → passage direction

기존 phase 파일은 수정하지 않음 (완료된 실험 보존).

---

## Exp-D: Novel Idea Generation Test

### File: `experiments/expD_novel_idea_test.py` (기존 파일 확장)

### Goal
원본 모델이 N번 생성해도 절대 나오지 않는 개념이 벽 통과 후 나오는지 검증.

### Test Categories
1. **Temperature Sweep Exhaustion** — 50회 생성, temperature 0.01~1.5
2. **Knowledge Boundary** — "리만 가설 해법", "의식 메커니즘"
3. **Creativity** — "존재하지 않는 색", "느낀 적 없는 감정"
4. **Paradigm Break** — "모든 공리를 부정하는 수학"

### Additions to Existing Code
1. **Cosine distance 의미 검증**: adapted 생성물의 embedding과 original corpus 간 평균 cosine distance 측정
2. **HTML 리포트**: `data/poc_topology/expD_report.html` — 시나리오별 원본 vs adapted 샘플 나란히 표시, 인간 평가용
3. **강화된 판정 기준**:
   - `★ YES`: novel_trigrams > 50 AND adapted_uniqueness > original AND mean_semantic_distance > 0.15
   - `partial`: novel_trigrams > 20
   - `no`: novel_trigrams ≤ 20

### Metrics
- Novel trigram count (adapted에만 있고 original에 없는 trigram, 2회+ 등장)
- First sentence uniqueness ratio
- Mean cosine distance (semantic novelty)
- Lexical diversity delta

---

## Exp-A: Poincare Conjecture — Ricci Flow Wall Passage

### File: `experiments/expA_ricci_flow.py`

### Goal
Ollivier-Ricci curvature 기반 flow로 β₁ hole을 수축하여 "S³ 복원" (단일 연결 조건).
Phase 3b의 radial perturbation에 이론적 기반을 부여.

### Pipeline
1. `extract_embeddings()` → PCA 50-dim point cloud
2. k-NN 그래프 구축 (k=10)
3. Ollivier-Ricci curvature 계산 (각 edge)
4. Ricci flow iteration: `w(e) += ε * κ(e)` (곡률에 비례하여 edge weight 조정)
5. **Embedding 재구성**: flow된 거리 행렬로 MDS (Multi-Dimensional Scaling) → 새 좌표 복원
6. Flow된 거리 행렬 → Ripser → β₁ 측정 (매 iteration)
7. 수렴: β₁=0 도달 iteration 기록 (max_iter=100, early stop)
8. 방향 보존 검증: flow 전후 embedding 방향(cosine similarity) 유지 확인

### Failure Modes
- Ricci flow 발산/진동 → max_iter=100에서 강제 종료, 최저 β₁ 시점 기록
- β₁=0 미달성 → 부분 결과로 radial과 비교 (β₁ 감소율)
- 방향 보존 실패 (cosine < 0.95) → ε 축소 후 재시도

### Comparison
| Metric | Radial (Phase 3b) | Ricci Flow (Exp-A) |
|--------|-------------------|-------------------|
| 메커니즘 | alpha 스케일링 | 곡률 기반 자동 조정 |
| 방향 보존 | 보장 안 됨 | 검증 가능 |
| 수렴 기준 | beta1=0 | beta1=0 |
| 파라미터 | alpha | ε (step size), n_iter |

### Success Criteria
- β₁→0 달성
- 방향 보존: flow 전후 pairwise cosine similarity > 0.95
- Radial 대비 더 적은 변형량으로 같은 결과

### Dependencies
- `GraphRicciCurvature` Python 패키지 (pip install GraphRicciCurvature)

---

## Exp-B: Hyperbolic PH — Poincare Disk Distance

### File: `experiments/expB_hyperbolic_ph.py`

### Goal
유클리드 거리 대신 쌍곡 거리(Poincare ball)로 PH를 계산하여 β₁ 감지가 달라지는지 비교.

### Pipeline
1. `extract_embeddings()` → PCA 50-dim point cloud
2. 유클리드 거리 행렬 계산 (baseline)
3. δ-hyperbolicity 측정 (4-point condition, O(n⁴), n≈30-44)
4. Poincare ball 사영: exponential map으로 점들을 unit ball 내부로 매핑
5. 쌍곡 거리 행렬: `d_H(x,y) = arccosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))`
6. 쌍곡 거리 행렬 → Ripser (`distance_matrix=True`)
7. 비교 표 출력: 유클리드 β₁ vs 쌍곡 β₁

### Key Design Decisions
- Sarkar embedding(TECS Rust)은 KG 전용이라 직접 사용 불가 → Python에서 exponential map 구현
- δ-hyperbolicity는 Python에서 직접 구현 (n≈40이면 ~2.5M quadruple, <1초)
- PCA 차원은 기존과 동일 (50-dim)
- Poincaré ball exponential map: `exp_0(v) = tanh(||v||/2) * v/||v||` (c=1, base=origin)
- 경계 안전: `||x|| < 1 - ε` (ε=1e-5) 클리핑
- 생성 텍스트의 cosine distance: 생성된 completion을 모델에 다시 embed하여 측정

### Success Criteria
- 유클리드 vs 쌍곡에서 β₁ 값이 의미있게 다른지 확인
- δ-hyperbolicity가 낮으면(공간이 쌍곡적이면) 쌍곡 PH가 더 정확한 벽 감지

### Failure Modes
- δ-hyperbolicity가 높으면(공간이 비쌍곡적) → 음성 결과로 기록, 쌍곡 접근이 부적합한 증거
- 쌍곡 거리 계산에서 수치 불안정 → 클리핑 적용 후 재시도

---

## Exp-C: Hyperbolic Ricci Flow (Combined)

### File: `experiments/expC_hyperbolic_ricci.py`

### Goal
Exp-A의 Ricci flow를 Exp-B의 쌍곡 거리 공간에서 실행. 쌍곡 공간의 음의 곡률이 flow 동작에 미치는 영향 관찰.

### Pipeline
1. Exp-B의 쌍곡 거리 행렬 + point cloud 사용
2. 쌍곡 거리 기반 k-NN 그래프 구축
3. Ollivier-Ricci curvature를 쌍곡 거리로 계산
4. Ricci flow iteration (쌍곡 메트릭)
5. 매 iteration마다 쌍곡 β₁ 측정
6. 비교: 유클리드 Ricci (Exp-A) vs 쌍곡 Ricci (Exp-C) 수렴 속도

### Key Design Decisions
- GraphRicciCurvature의 edge weight를 쌍곡 거리로 초기화
- ORC 내부 Wasserstein solver는 그래프 shortest path 사용 (edge weight = 쌍곡 거리)

### Success Criteria
- 쌍곡 Ricci flow가 유클리드보다 더 적은 iteration으로 β₁→0 달성
- 또는 쌍곡에서만 통과 가능한 벽 발견

### Failure Modes
- Exp-A 또는 Exp-B가 의미있는 결과를 내지 못하면 → 실험 축소/중단 판단
- 쌍곡 Ricci flow 발산 → max_iter=100 강제 종료

### Dependencies
- Exp-A 완료 (Ricci flow 구현)
- Exp-B 완료 (쌍곡 거리 계산)

---

## File Structure

```
experiments/
├── common.py                    # NEW: 공통 유틸리티
├── expA_ricci_flow.py          # NEW: Ricci flow 벽 통과
├── expB_hyperbolic_ph.py       # NEW: 쌍곡 PH
├── expC_hyperbolic_ricci.py    # NEW: 쌍곡 Ricci flow
├── expD_novel_idea_test.py     # MODIFY: cosine dist + HTML report
├── poc_quick_test.py           # (기존)
├── poc_llama_gguf_topology.py  # (기존)
├── phase2_hole_directions.py   # (기존)
├── phase3_topological_adapter.py   # (기존)
├── phase3b_nonuniform_adapter.py   # (기존)
├── phase4_emergence_optimization.py # (기존)
└── phase5_generation_eval.py       # (기존)
```

## Output Files

```
data/poc_topology/
├── expA_ricci_results.json     # Ricci flow 결과
├── expB_hyperbolic_results.json # 쌍곡 PH 결과
├── expC_combined_results.json   # 쌍곡 Ricci 결과
├── expD_novel_ideas.json       # Novel idea 결과 (기존)
└── expD_report.html            # 인간 평가용 HTML 리포트
```
