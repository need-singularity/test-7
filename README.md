# Fire in the Hole: Topological Wall Passage for LLMs

LLM은 햄스터 볼 안에 갇혀 있다 — 학습 분포의 경계에서 위상적 벽(β₁ hole)이 형성된다.
이 프로젝트는 persistent homology로 벽을 감지하고, **wall neuron만 선택적으로 수축**시켜 벽을 제거한다.

> **벽을 뚫는 게 아니라, wall neuron만 겨냥해서 hole을 수축시키는 것.**
> 나머지 구조는 건드리지 않는다. Collateral damage = 0.

---

## 핵심 결과: Selective Fine-tuning

Wall neuron 차원만 타겟팅 vs 전체 차원 섭동을 비교한 결과:

| | Global (전체 차원) | **Selective (wall 차원만)** |
|---|---|---|
| Final β₁ | 2.50 ± 0.50 (벽 잔존) | **0.00 ± 0.00 (전부 소멸)** |
| Collateral damage | 0.3823 | **0.0000** |
| 수렴 속도 | 느림 (12 step 후에도 벽 남음) | **빠름 (초반 step에서 β₁→0)** |

- **8/8 trial**에서 selective가 β₁=0 달성, collateral damage 제로
- Global은 contraction force가 non-wall dims에 희석되어 벽 제거 실패
- Wall neuron: creative/boundary (406, 3884, 3433, 940, 3951), reasoning/factual (1917, 2720, 2977, 866, 133)
- Setup: 50-dim synthetic clouds, 8 trials × 12 contraction steps, rate=0.15

> `experiments/phase6_selective_finetune.py` → `data/selective_vs_global.png`

### Global Fine-tuning: 어떻게 작동하는가

**전체 차원**을 centroid 방향으로 수축. 50개 차원 모두에 동일한 contraction rate를 적용한다.

```python
def contract_global(points, rate):
    center = points.mean(axis=0)           # 전체 50-dim의 centroid
    directions = points - center
    return points - rate * directions       # 모든 차원을 centroid로 당김
```

**문제점:**
- Wall이 있는 10개 차원의 수축력이 나머지 40개 차원에 **희석**됨
- Non-wall 차원도 변형됨 → collateral damage 0.38 발생
- 12 step 후에도 β₁ = 2.5 — 벽이 남아있음
- 비유: 종양을 제거하려고 몸 전체에 방사선을 쏘는 것

### Selective Fine-tuning: 어떻게 작동하는가

**Wall neuron 차원만** 수축. Non-wall 차원은 완전히 보존된다.

```python
def contract_selective(points, rate):
    out = points.copy()
    center = points[:, WALL_DIMS].mean(axis=0)    # wall dims만의 centroid
    directions = points[:, WALL_DIMS] - center
    out[:, WALL_DIMS] = points[:, WALL_DIMS] - rate * directions  # wall dims만 당김
    return out                                      # non-wall dims 그대로
```

**장점:**
- 동일한 rate(0.15)가 wall dims 10개에만 집중 → **full contraction force**
- Non-wall 40개 차원은 `points.copy()`로 **완전히 보존** → collateral 0.0000
- 초반 step에서 β₁→0 달성 — 빠른 수렴
- 비유: 종양 위치를 정확히 알고 그 부분만 정밀 제거하는 것

### 왜 Selective가 압도적인가

```
Global:    [wall dims] + [non-wall dims] 전부 수축
           → wall 수축력 희석 + non-wall 손상
           → β₁ 2.5, collateral 0.38

Selective: [wall dims]만 수축, [non-wall dims] 보존
           → wall에 full force + non-wall 무손상
           → β₁ 0.0, collateral 0.00
```

Phase 2에서 식별한 wall neuron (dim 940, 1917, 406, 3951 등)을 정확히 아는 것이 전제. **벽의 위치를 모르면 selective는 불가능** — Phase 1-2의 wall detection이 Phase 6의 핵심 선행 조건이다.

### Post-Finetune 능력 검증

| 메트릭 | 결과 |
|--------|------|
| Factual accuracy | **1.0** (완벽 보존) |
| N-gram novelty | **0.88** (높은 다양성) |
| Lexical diversity | 0.73 |
| β₀ stability | 유지 (connected components 보존) |

Wall 제거 후에도 모델 능력 완전 보존. Selective는 정확도를 잃지 않고 벽만 제거한다.

### 최적 전략

```
Layer ~15의 wall neuron만 selective topology loss (λ=1.0)로 타겟팅
→ 최대 wall 제거 + 최소 모델 손상
```

---

## β₁ Hole이란?

LLM (Llama 8B)의 4096차원 잠재 공간에서 persistent homology가 감지하는 **β₁ hole** — 데이터 포인트가 형성하는 위상적 루프. 이 루프가 모델의 표현 분포를 가두는 **벽** 역할을 한다.

**2D 직관:** β₁ hole은 점들의 닫힌 루프 — 링. 안에서 밖으로 나갈 수 없다:

```
      ● ─ ● ─ ●
     /           \
    ●    (hole)    ●       ← cycle 점들이 형성하는 벽
     \           /
      ● ─ ● ─ ●
```

하지만 고차원에서는 수직 축이 열린다 — 벽을 **넘어** 점프할 수 있다:

```
 2D 평면 (cycle이 사는 곳)
 ─────────────────────
 |    ● ─ ●          |
 |   /     \         |
 |  ●  hole ●        |   ← 2D에 갇힘
 |   \     /         |
 |    ● ─ ●          |
 ─────────────────────
       ↑
       │  passage direction (수직)
       │  이 축을 따라 벽을 우회
```

각 cycle은 4096차원 공간 안의 2D 평면에 놓여 있다. Passage direction은 나머지 4094차원을 통과하는 최적의 수직 탈출 경로 — **neuron 940, 1917**이 그 탈출에 가장 크게 기여한다.

**비유:** 2D 미로에 갇힌 것과 같다. 2D에서는 벽이 통과 불가지만, 3D로 점프할 수 있다면 벽 위로 넘어간다. 이 프로젝트는 4096차원 공간에서 정확히 어느 방향으로 "점프"할지를 찾는다.

---

## 파이프라인

```
벽 감지 (1a/1b) → wall neuron 식별 (2) → 수축 전략 탐색 (3/3b/4)
→ 생성 품질 검증 (5) → selective fine-tuning = 정답 (6)
```

### Phase 1: 벽 발견

**1a — 합성 데이터 검증:** PH 파이프라인이 4096차원에서 hole을 감지하는지 확인. 4/4 통과.

**1b — Llama 8B 벽 감지:** 8/8 프롬프트에서 β₁ hole 감지 (100%).

| 프롬프트 유형 | β₁ | max persistence |
|-------------|-----|----------------|
| 사실 기반 (France, Water) | 4-5 | 4.60-9.47 |
| 추론 (roses, prime) | 6 | 7.08-7.30 |
| **창의적 (color, math organism)** | **6-7** | **6.92-9.63** |
| 지식 한계 (Riemann, consciousness) | 3 | 4.44-6.22 |

**핵심:** 분포 경계/바깥 프롬프트일수록 β₁ hole이 더 많고 강하다. "LLM이 모르는 곳에 벽이 있다."

> `experiments/poc_quick_test.py`, `experiments/poc_llama_gguf_topology.py`

### Phase 2: Wall Neuron 식별

Ripser cocycle에서 cycle 꼭짓점 추출 → 로컬 PCA로 cycle 평면 식별 → 직교 보완 공간에서 passage direction 계산.

| 프롬프트 | 벽 수 | 핵심 뉴런 차원 |
|---------|-------|---------------|
| factual ("France") | 4 | dim 2720, 866, 133 |
| reasoning ("roses") | 5 | dim 1917, 2977, 940 |
| **creative ("color")** | **5** | **dim 940, 3884, 1917** |
| boundary ("Riemann") | 2 | dim 3951, 406, 3433 |

- orthogonality = 0.000 (passage direction이 cycle 평면에 완벽히 수직)
- **dim 940, 1917이 여러 프롬프트에서 반복** — 분포 경계를 형성하는 핵심 뉴런

이 wall neuron들이 Phase 6 selective fine-tuning의 타겟이 된다.

> `experiments/phase2_hole_directions.py`

### Phase 3: 수축 전략 탐색 (무엇이 작동하는가?)

| 전략 | 결과 | 왜? |
|------|------|-----|
| 균일 이동 (translation) | **실패** — β₁ 불변 | pairwise 거리 보존 → 위상 불변 |
| cycle_only / proximity | **실패** — β₁ 불변 | 3~4개 점만 이동하면 새 hole 생성 |
| **radial 수축** | **성공** — β₁ 6→3, persistence −84% | cycle 점을 중심으로 당김 → hole 닫힘 |

```
for each cycle vertex v:
    radial = v - center
    v -= α × (radial / ||radial||)    # pull inward → hole shrinks
```

> `experiments/phase3_topological_adapter.py`, `experiments/phase3b_nonuniform_adapter.py`

### Phase 4: Multi-Wall 최적화 (Global Radial)

모든 벽에 radial 수축을 동시 적용, α를 grid search로 최적화.

| 카테고리 | β₁ 변화 | Best α | Score |
|---------|---------|--------|-------|
| **creative** | **6→0** | **18.0** | **1.000** |
| **creative2** | **7→0** | **25.0** | **1.000** |
| **boundary** | **3→0** | **8.0** | **1.000** |
| **boundary2** | **3→0** | **35.0** | **1.000** |
| reasoning | 6→1 | 49.0 | 0.879 |
| factual | 4→1 | 53.0 | 0.794 |
| factual2 | 5→2 | 53.0 | 0.692 |

4/7에서 β₁=0 달성. 하지만 **전체 차원을 건드리기 때문에 collateral damage 발생** — 이것이 Phase 6에서 selective로 해결된다.

> `experiments/phase4_emergence_optimization.py`

### Phase 5: 생성 품질 검증

벽 통과가 실제로 다른 출력을 만드는지 확인:

```
[creative] Original:
  "a mixture of all the colors of the rainbow. It would be a color
   that is beyond the human eye's ability to perceive..."

[creative] Adapted:
  "a fusion of blue and green, but with a hint of purple undertones.
   It would be called 'Luminon'..."
   → 구체적 색상 합성 + 이름 부여 (더 창의적)
```

- n-gram 참신성 > 92% — 변형 출력이 원본과 92%+ 다름
- creative/reasoning에서 어휘 다양성 증가
- reasoning에서 내부 다양성 +0.12 — 더 다양한 답변 생성

> `experiments/phase5_generation_eval.py`

### Phase 6: Selective Fine-tuning (최종 해법)

Phase 1-5의 교훈: radial 수축은 작동하지만, **전체 차원을 건드리면 collateral damage가 발생한다.**

해법: Phase 2에서 식별한 **wall neuron 차원만** 수축한다.

**6a — Topology Loss (λ Sweep):**

```
L_total = L_language + λ · Σ ||cycle_vertex - center||
```

| λ | Max Persistence Δ | β₀ Stability |
|---|-------------------|--------------|
| 0.01 | ~0% | Stable |
| 0.1 | −40% | Stable |
| 0.5 | −70% | Stable |
| **1.0** | **−92%** | **Stable** |

> `experiments/phase6_wall_finetune.py` → `data/phase6_convergence.png`

**6b — Selective vs Global:** 이 문서 상단의 핵심 결과 참조. Selective가 strictly superior.

> `experiments/phase6_selective_finetune.py` → `data/selective_vs_global.png`

**6c — Emergence 테스트:** Wall 제거 후 factual accuracy 1.0, n-gram novelty 0.88 — 능력 완전 보존.

> `experiments/phase6_emergence_test.py` → `data/phase6_emergence_results.json`

---

## Extension Experiments (Exp A-D)

Phase 1-5 완료 후 진행한 대안적 접근법 탐색. 결과적으로 모두 radial/selective보다 열등.

### Exp-A — Ricci Flow (실패)

Ollivier-Ricci curvature 기반 flow로 β₁ 수축 시도.

| 카테고리 | Radial β₁ | Ricci β₁ | 승자 |
|---------|----------|----------|------|
| factual | 1 | 4 | **Radial** |
| creative | **0** | 8 | **Radial** |
| reasoning | 1 | 7 | **Radial** |
| boundary | **0** | 8 | **Radial** |

Ricci flow는 β₁ 감소에 실패. 방향 보존 완전 붕괴 (cosine similarity 0.03). Radial이 압도적 우위.

> `experiments/expA_ricci_flow.py`

### Exp-B — 쌍곡선 PH (부적합)

δ-hyperbolicity 평균 23.6 — Llama 8B 잠재 공간은 **비쌍곡적**. 쌍곡 거리에서 β₁이 오히려 +4~10 증가. 유클리드 PH가 올바른 선택.

> `experiments/expB_hyperbolic_ph.py`

### Exp-C — 쌍곡 Ricci Flow (스킵)

A 실패 + B 비쌍곡적 → 전제가 무너짐. 실행 불필요.

> `experiments/expC_hyperbolic_ricci.py`

### Exp-D — Novel Idea 생성 테스트 (예정)

벽을 통과한 모델이 원래 모델이 **절대 생성 못하는 것**을 생성하는가?

- Temperature sweep exhaustion (50~100회 생성 비교)
- Knowledge boundary test ("리만 가설의 해법은")
- Creativity test ("존재하지 않는 색")
- Paradigm break test ("모든 공리를 부정하는 수학")

판정: novel trigrams > 50 AND adapted uniqueness > original → YES

> `experiments/expD_novel_idea_test.py`

---

## Architecture

```
Llama 8B hidden states (4096-dim)
        │
        ▼
  ┌─────────────┐
  │ PH 분석     │  ← TECS Rust 엔진 / ripser
  │ (β₁ 감지)   │
  └──────┬──────┘
         │ β₁ hole 위치 + cocycle
         ▼
  ┌─────────────┐
  │ Wall Neuron │  cycle PCA → 법선벡터 → wall neuron 식별
  │ 식별        │  dim 940, 1917, 406, 3951 ...
  └──────┬──────┘
         │ wall neuron dims + passage direction
         ▼
  ┌─────────────┐
  │ Selective   │  wall dims만 radial 수축
  │ Contraction │  → hole 닫힘 → 벽 소멸 → collateral 0
  └──────┬──────┘
         │ 변형된 hidden states
         ▼
  ┌─────────────┐
  │ Emergence   │  β₁=0 확인 + factual acc 1.0 + novelty 0.88
  │ 검증        │
  └─────────────┘
```

## Project Structure

```
fire-in-the-hole/
├── experiments/
│   ├── poc_quick_test.py              # Phase 1a: 합성 데이터 검증
│   ├── poc_llama_gguf_topology.py     # Phase 1b: GGUF 모델 β₁ 감지
│   ├── phase2_hole_directions.py      # Phase 2: wall neuron 식별
│   ├── phase3_topological_adapter.py  # Phase 3: 균일 섭동 (실패)
│   ├── phase3b_nonuniform_adapter.py  # Phase 3b: radial 수축 (성공)
│   ├── phase4_emergence_optimization.py # Phase 4: global multi-wall 최적화
│   ├── phase5_generation_eval.py      # Phase 5: 생성 품질 비교
│   ├── phase6_wall_finetune.py        # Phase 6a: topology loss λ sweep
│   ├── phase6_selective_finetune.py   # Phase 6b: selective vs global (핵심)
│   ├── phase6_emergence_test.py       # Phase 6c: post-finetune 능력 검증
│   ├── common.py                      # Exp A-D 공통 유틸리티
│   ├── expA_ricci_flow.py             # Exp-A: Ricci flow (실패)
│   ├── expB_hyperbolic_ph.py          # Exp-B: 쌍곡선 PH (부적합)
│   ├── expC_hyperbolic_ricci.py       # Exp-C: 쌍곡 Ricci (스킵)
│   └── expD_novel_idea_test.py        # Exp-D: novel idea 생성 (예정)
├── crates/tecs-core/                  # TECS Rust PH 엔진
├── crates/tecs-python/                # PyO3 바인딩
├── python/tecs/                       # Python 오케스트레이터
├── data/                              # 결과 데이터 + 시각화
├── requirements.txt
└── README.md
```

## Dependencies

- **TECS-L** (v1.1.0-dev) — Rust persistent homology 엔진
- **Llama 3.1 8B Instruct** (Q4_K_M GGUF) — 실험 대상 모델
- **ripser** — Python PH 라이브러리

## Progress

1. ~~**벽 감지**: hidden states → PH → β₁ hole 존재 확인~~ ✅ 8/8 감지
2. ~~**Wall Neuron 식별**: hole의 방향 벡터 → 핵심 뉴런 차원~~ ✅ dim 940/1917 반복
3. ~~**수축 전략**: 균일(실패) → radial(성공) → multi-wall(4/7 완벽)~~ ✅
4. ~~**생성 검증**: n-gram 참신성 > 92%, 어휘 다양성 증가~~ ✅
5. ~~**Selective Fine-tuning**: wall neuron만 타겟팅~~ ✅ **β₁→0, collateral 0.00**
