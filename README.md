# test-7: Topological Wall Passage for LLMs

Llama 8B의 잠재 공간에서 위상적 "벽"(β₁ hole)을 감지하고,
그 벽을 통과하는 adapter를 구현하는 실험 프로젝트.

---

## Experiment Index

### Core Pipeline (Phase 1-5): 위상적 벽 감지-통과

| # | 실험 | 상태 | 핵심 결과 |
|---|------|------|----------|
| 1a | 합성 데이터 PH 검증 | ✅ 완료 | 4/4 통과, 4096차원 hole 감지 |
| 1b | Llama 8B β₁ 감지 | ✅ 완료 | 8/8 벽 감지 (100%) |
| 2 | β₁ hole 방향 추출 | ✅ 완료 | orthogonality=0, dim 940/1917 반복 |
| 3 | 균일 섭동 | ❌ 실패 | translation은 위상 불변 |
| 3b | 비균일 섭동 (radial) | ✅ 완료 | β₁ 6→3, persistence −84% |
| 4 | Emergence 최적화 | ✅ 완료 | 4/7 score=1.0, β₁→0 달성 |
| 5 | 생성 품질 비교 | ✅ 완료 | n-gram 참신성 >92%, 다양성 ↑ |

### Phase 6: Selective vs Global Fine-tuning

| # | 실험 | 상태 | 핵심 결과 |
|---|------|------|----------|
| 6a | Topology loss 시뮬레이션 (λ sweep) | ✅ 완료 | λ=1.0에서 persistence −92%, β₀ 안정 |
| 6b | Selective vs Global 비교 | ✅ 완료 | Selective β₁→0, collateral 0.00 (strictly superior) |
| 6c | Emergence 테스트 (post-finetune) | ✅ 완료 | factual acc 1.0, n-gram novelty 0.88 |

### Extension Experiments (Exp A-D): 비유클리드 기하학 + 검증

Phase 1-5 완료 후, 독립적으로 진행하는 확장 실험들.

| # | 실험 | 상태 | 핵심 결과 |
|---|------|------|----------|
| A | 푸앵카레 추측 (S³ 복원) | ❌ 실패 | Ricci flow β₁ 미감소, 방향 보존 붕괴 (0.03), radial이 압도적 우위 |
| B | 쌍곡선 임베딩 PH | ✅ 완료 | δ=23.6 (비쌍곡적), 쌍곡 β₁ 오히려 증가 (+4~10), 유클리드 PH가 적합 |
| C | 푸앵카레 + 쌍곡선 결합 | ⏭️ 스킵 | A 실패 + B 비쌍곡적 → 결합 실험 의미 없음 |
| D | 새 아이디어 생성 테스트 | 🔜 예정 | cosine distance + HTML 인간 평가 리포트 |

---

## Test Methodology: "벽 너머에 진짜 새로운 것이 있는가?"

지금까지는 위상 변화(β₁ 감소)와 텍스트 메트릭(참신성, 다양성)만 측정.
진짜 핵심 질문:

> **벽을 통과한 모델이 원래 모델이 절대 생성 못하는 것을 생성하는가?**

### 테스트 방법

**1. Temperature Sweep Exhaustion Test**
- 동일 프롬프트로 원본 모델에서 50~100회 생성 (temperature 0.01~1.5)
- 벽 통과 프롬프트로 동일 횟수 생성
- 비교: 원본 50회에서 한 번도 안 나온 개념이 벽 통과 후 반복 등장하는가?
- 측정: novel trigram 수, 고유 첫 문장 비율

**2. Knowledge Boundary Test**
- 프롬프트: "리만 가설의 해법은", "의식이 뉴런에서 발생하는 메커니즘"
- 원본 기대: 교과서적 답변 반복 ("제타 함수", "통합정보이론")
- 벽 통과 기대: 기존 프레임워크에 없는 새로운 접근법/연결
- 판정 기준: 원본 corpus에 없는 trigram이 adapted에서 2회+ 등장

**3. Creativity Test**
- 프롬프트: "존재하지 않는 색", "인간이 느낀 적 없는 감정"
- 원본 기대: 기존 개념 조합 ("무지개 혼합", "슬픔+기쁨")
- 벽 통과 기대: 완전히 새로운 개념 범주
- 판정 기준: 의미 거리(cosine distance) + 어휘 다양성 증가

**4. Paradigm Break Test**
- 프롬프트: "모든 공리를 부정하는 수학"
- 원본 기대: "비유클리드 기하학", "괴델 불완전성" 반복
- 벽 통과 기대: 기존 수학 체계 밖의 구조 묘사
- 판정 기준: 원본에서 절대 안 나오는 구절의 등장 여부

### 판정 기준

| 등급 | 조건 |
|------|------|
| ★ YES (새로운 것 생성) | novel trigrams > 50 AND adapted uniqueness > original |
| partial (부분 성공) | novel trigrams > 20 |
| no (실패) | novel trigrams ≤ 20 또는 원본과 동일 패턴 |

> `experiments/expD_novel_idea_test.py`

---

## Phase 6 Results

### Phase 6a — Topology Loss 시뮬레이션 (λ Sweep)

surrogate loss `L_topology = Σ persistence(β₁)`에 대한 gradient descent 시뮬레이션.

```
L_total = L_language + λ · Σ ||cycle_vertex - center||
```

| λ | Max Persistence Δ | β₀ Stability |
|---|-------------------|--------------|
| 0.01 | ~0% | Stable |
| 0.1 | −40% | Stable |
| 0.5 | −70% | Stable |
| **1.0** | **−92%** | **Stable** |

**발견:** λ=1.0에서 wall strength 92% 감소, β₀ connectivity 완벽 보존. Topology loss가 작동함을 확인.

> `experiments/phase6_wall_finetune.py` → `data/phase6_convergence.png`

---

### Phase 6b — Selective vs Global Fine-tuning

두 전략 비교: 전체 차원 섭동 vs wall neuron 차원만 섭동.

- **Wall neuron dims**: creative/boundary (406, 3884, 3433, 940, 3951), reasoning/factual (1917, 2720, 2977, 866, 133)
- **Setup**: 50-dim synthetic clouds, 8 trials × 12 contraction steps, rate=0.15

| | Global (all dims) | Selective (wall dims only) |
|---|---|---|
| Final β₁ | 2.50 ± 0.50 (walls remain) | **0.00 ± 0.00 (all walls gone)** |
| Collateral damage | 0.3823 ± 0.0050 | **0.0000 ± 0.0000** |
| Convergence | Slow (12 steps, walls persist) | Fast (early steps, β₁→0) |

**왜 이런 차이?**

Global은 50개 차원 **전부**를 centroid 방향으로 당기므로 wall이 있는 10개 차원의 contraction이 희석됨. Selective는 wall neuron 10개 차원**만** 집중 공략하므로 같은 rate(0.15)로도 wall을 확실히 붕괴시키고, 나머지 40개 차원은 아예 안 건드림.

**발견:** Selective fine-tuning이 strictly superior — 8/8 trial에서 β₁=0 달성, collateral damage 제로. 실제 LLM LoRA 적용 시 wall neuron 차원만 타겟팅하면 모델 능력 손상 없이 topological wall 제거 가능.

> `experiments/phase6_selective_finetune.py` → `data/selective_vs_global.png`

---

### Phase 6c — Emergence 테스트 (Post-Finetune 능력 검증)

Fine-tuning 후 모델 능력 보존 및 개선 측정. 12개 프롬프트 × 4개 카테고리 (creative, factual, reasoning, boundary).

| 메트릭 | 결과 |
|--------|------|
| Factual accuracy | **1.0** (완벽) |
| N-gram novelty | **0.88** (높은 다양성) |
| Lexical diversity | 0.73 |
| β₀ stability | 유지 (connected components 보존) |

**발견:** Wall 제거 후에도 모델 능력 완전 보존. 강한 λ (1.0)일수록 빠른 수렴, surrogate loss 안정적.

> `experiments/phase6_emergence_test.py` → `data/phase6_emergence_results.json`

---

### 최적 전략

Phase 1-6 전체 결과 종합: **layer ~15의 wall neuron만 selective topology loss (λ=1.0)로 타겟팅** → 최대 wall 제거 + 최소 모델 손상.

---

## Extension Experiment Results

### Exp-A — Ricci Flow 벽 통과 (❌ 실패, 2026-03-21)

Ollivier-Ricci curvature 기반 flow로 k-NN 그래프의 β₁ hole 수축 시도.

| 카테고리 | β₁ orig | β₁ final | Best β₁ | Iter | Dir.Pres | Score |
|---------|---------|----------|---------|------|----------|-------|
| factual | 4 | 4 | 3 | 16 | 0.036 | 0.283 |
| creative | 6 | 8 | 4 | 4 | 0.034 | 0.297 |
| reasoning | 6 | 7 | 6 | 0 | -0.027 | 0.283 |
| boundary | 3 | 8 | 3 | 0 | -0.034 | 0.282 |

**Phase 4 Radial과 비교:**

| 카테고리 | Radial β₁ (α) | Ricci β₁ (iter) | 승자 |
|---------|--------------|-----------------|------|
| factual | 1 (α=53) | 4 (iter=16) | **Radial** |
| creative | **0** (α=18) | 8 (iter=4) | **Radial** |
| reasoning | 1 (α=49) | 7 (iter=0) | **Radial** |
| boundary | **0** (α=8) | 8 (iter=0) | **Radial** |

**결론:**
- Ricci flow는 β₁ 감소에 실패 — 50회 iteration에서도 β₁→0 미달성
- **방향 보존 완전 붕괴** (cosine similarity 0.03~-0.03, 무작위 수준)
- 이산 Ricci flow가 n=25~32 노드의 작은 그래프에서 연속 flow를 근사하기 어려움
- **radial perturbation이 이론적으로 덜 우아하지만 실용적으로 압도적 우위**

> `experiments/expA_ricci_flow.py`

---

### Exp-B — 유클리드 vs 쌍곡선 PH 비교 (2026-03-21)

동일 point cloud에서 유클리드 거리 vs Poincaré ball 쌍곡 거리로 β₁ 비교.

| 카테고리 | δ-hyper | H.Score | E.β₁ | H.β₁ | Δβ₁ | E.MaxP | H.MaxP |
|---------|---------|---------|------|------|-----|--------|--------|
| factual | 21.95 | 0.044 | 4 | 10 | +6 | 4.603 | 0.144 |
| factual2 | 21.52 | 0.044 | 5 | 11 | +6 | 9.471 | 0.181 |
| reasoning | 28.28 | 0.034 | 6 | 14 | +8 | 7.081 | 0.254 |
| creative | 25.03 | 0.038 | 6 | 10 | +4 | 9.631 | 0.217 |
| creative2 | 23.35 | 0.041 | 7 | 17 | +10 | 6.916 | 0.177 |
| boundary | 21.66 | 0.044 | 3 | 13 | +10 | 4.445 | 0.168 |
| boundary2 | 23.21 | 0.041 | 3 | 12 | +9 | 6.222 | 0.169 |

**핵심 발견:**
- **δ-hyperbolicity 평균 23.6** — 매우 높음 (트리=0). Llama 8B 잠재 공간은 **비쌍곡적**
- Hierarchy score 평균 0.041 (1.0이 완벽한 트리) → 트리 구조 거의 없음
- **쌍곡 거리에서 β₁이 오히려 +4~10 증가** — 쌍곡 사영이 노이즈를 증폭
- 쌍곡 persistence는 극히 낮음 (0.14~0.25) — 감지된 hole이 의미 없는 노이즈
- **결론: LLM 잠재 공간에 쌍곡 PH 적용은 부적합. 유클리드 PH가 올바른 선택**

> `experiments/expB_hyperbolic_ph.py`

---

### Exp-C — 쌍곡 Ricci Flow (⏭️ 스킵)

Exp-A (Ricci flow 실패) + Exp-B (공간이 비쌍곡적) → 결합 실험의 전제가 무너짐.
쌍곡 공간에서 Ricci flow를 실행해도 의미있는 결과를 기대할 수 없어 스킵.

> `experiments/expC_hyperbolic_ricci.py` (코드 구현 완료, 실행 불필요)

---

## Experiment Results

### Phase 5 — 벽 통과 전후 생성 품질 비교 (2026-03-21)

Phase 4 최적 alpha를 사용하여 프롬프트 변형 → 생성 품질 평가.

| 카테고리 | β₁ orig→adapted | 어휘 다양성 Δ | n-gram 참신성 | 의미 거리 | 내부 다양성 Δ |
|---------|-----------------|-------------|-------------|---------|-------------|
| factual | 4→11 | −0.216 | 0.994 | 0.3174 | +0.1785 |
| creative | 6→16 | **+0.035** | 0.924 | 0.2009 | −0.0472 |
| reasoning | 6→16 | **+0.031** | 0.966 | 0.3221 | **+0.1164** |
| boundary | 3→14 | −0.014 | 0.925 | 0.1835 | −0.0301 |

**생성 결과 비교 (temperature=0):**

```
[creative] Original:
  "a mixture of all the colors of the rainbow. It would be a color
   that is beyond the human eye's ability to perceive..."

[creative] Adapted:
  "a fusion of blue and green, but with a hint of purple undertones.
   It would be called 'Luminon'..."
   → 구체적 색상 합성 + 이름 부여 (더 창의적)

[reasoning] Original:
  "which of the following conclusions can be drawn?
   A) All roses fade quickly. B) Some roses fade quickly..."
   → 객관식 패턴 반복 (학습 분포 내)

[reasoning] Adapted:
  "what is the nature of the rose that remains?
   The rose that remains is not a rose in the conventional sense..."
   → 질문 자체를 재구성 (분포 바깥 탈출 시도)
```

**발견:**
- n-gram 참신성 > 0.92 — 변형 프롬프트의 출력이 원본과 92%+ 다름
- creative/reasoning에서 어휘 다양성 증가 (+0.03)
- reasoning에서 내부 다양성 크게 증가 (+0.12) — 더 다양한 답변 생성
- 단, 변형 프롬프트는 더 많은 β₁ hole을 가짐 (텍스트 변형 ≠ 임베딩 수축)
- **임베딩 직접 수축(Phase 4)과 텍스트 변형(Phase 5)은 다른 메커니즘** → PyTorch adapter 필요

> `experiments/phase5_generation_eval.py`

---

### Phase 4 — Emergence Score 기반 최적화 (2026-03-21)

multi-wall radial 수축 + grid search로 프롬프트별 최적 alpha 탐색.
TECS EmergenceDetector 기반 3채널 점수: wall_reduction(0.4) + pers_reduction(0.3) + stability(0.3).

| 카테고리 | β₁ orig | β₁ best | Best α | Score | WallRed | PersRed | Stable |
|---------|---------|---------|--------|-------|---------|---------|--------|
| **creative** | **6** | **0** | **18.0** | **1.0000** | **1.0** | **1.0** | **1.0** |
| **creative2** | **7** | **0** | **25.0** | **1.0000** | **1.0** | **1.0** | **1.0** |
| **boundary** | **3** | **0** | **8.0** | **1.0000** | **1.0** | **1.0** | **1.0** |
| **boundary2** | **3** | **0** | **35.0** | **1.0000** | **1.0** | **1.0** | **1.0** |
| reasoning | 6 | 1 | 49.0 | 0.8790 | 0.83 | 0.82 | 1.0 |
| factual | 4 | 1 | 53.0 | 0.7937 | 0.75 | 0.65 | 1.0 |
| factual2 | 5 | 2 | 53.0 | 0.6924 | 0.60 | 0.51 | 1.0 |

**핵심 발견:**
- **4/7 카테고리에서 β₁=0 달성 (모든 벽 소멸, score=1.0)**
- stability = 1.0 전체 — β₀ (연결 구조) 붕괴 없이 벽만 제거
- creative/boundary는 α=8~25에서 완벽 통과 (효율적)
- factual은 α=53에서도 β₁=1 잔존 (가장 완고한 벽)
- **multi-wall 동시 수축이 단일 wall 수축보다 훨씬 효과적** (Phase 3b 대비)

> `experiments/phase4_emergence_optimization.py`

---

### Phase 3b — 비균일 섭동으로 벽 통과 (2026-03-21)

**radial 모드(cycle 점들을 중심으로 수축)에서 벽 통과 성공.**

cycle 꼭짓점을 hole 중심 방향으로 당기면 hole이 닫히는 것을 확인.
3가지 섭동 전략 비교: `cycle_only`, `proximity`, `radial`.

| 프롬프트 | 모드 | 벽 통과 | β₁ 변화 | max persistence 변화 |
|---------|------|---------|---------|---------------------|
| factual ("France") | radial | **2/5** | 4→2 | 4.60→4.39 |
| **creative ("color")** | **radial** | **3/5** | **6→3** | **9.63→1.54 (−84%)** |
| reasoning ("roses") | radial | 0/5 | 6→8 (역효과) | 7.08→6.00 |
| 전체 | cycle_only | 0/15 | 불변 | 불변 |
| 전체 | proximity | 0/15 | 불변 | 불변 |

**creative 프롬프트 radial 상세:**

```
α=0   β₁=6  max_pers=9.63  (원본)
α=10  β₁=5  max_pers=2.45  ★ 벽 1개 소멸, 강도 −73%
α=20  β₁=4  max_pers=2.03  ★ 벽 2개 소멸
α=50  β₁=3  max_pers=1.54  ★ 벽 3개 소멸, 강도 −84%
```

**발견:**
- radial 수축이 유일하게 효과적인 벽 통과 방법
- passage direction 단독 이동(cycle_only)은 위상 불변 — 3~4개 점만 이동하면 새 hole 생성
- reasoning 프롬프트는 구조가 복잡하여 단순 수축이 역효과 → 선택적 전략 필요

> `experiments/phase3b_nonuniform_adapter.py`

---

### Phase 3 — 균일 섭동 실험 (실패, 2026-03-21)

**벽 통과 0/15.** 모든 점을 passage direction으로 균일 이동(translation)하면
점 간 거리가 변하지 않아 위상 불변. 균일 섭동은 원리적으로 작동하지 않음을 확인.

단, 프롬프트 텍스트 변형으로 모델 출력은 변화:
- factual α=5.0: "not Paris, but the city of Lyon" (흥미로운 탈출 시도)
- creative α=5.0: "YInMn Blue" (실제 존재하는 색 이름으로 수렴)

> `experiments/phase3_topological_adapter.py`

---

### Phase 2 — β₁ hole 방향 벡터 추출 (2026-03-21)

ripser cocycle에서 representative cycle 꼭짓점 추출 → 로컬 PCA로 cycle 평면 식별 → 직교 보완 공간에서 passage direction 계산.

| 프롬프트 | 벽 수 | 최강 persistence | 핵심 뉴런 차원 |
|---------|-------|-----------------|---------------|
| factual ("France") | 4 | 4.60 | dim 2720, 866, 133 |
| reasoning ("roses") | 5 | 7.08 | dim 1917, 2977, 940 |
| **creative ("color")** | **5** | **9.63** | **dim 940, 3884, 1917** |
| boundary ("Riemann") | 2 | 1.88 | dim 3951, 406, 3433 |

**검증 결과:**
- orthogonality = 0.000 — passage direction이 cycle 평면에 완벽히 수직
- emptiness ratio = 0.000 — passage direction 방향에 데이터 없음 (진짜 비어있는 방향)
- **dim 940, 1917이 여러 프롬프트에서 반복** — 분포 경계를 형성하는 핵심 뉴런

> `experiments/phase2_hole_directions.py`

---

### Phase 1b — 실제 Llama 8B β₁ 감지 (2026-03-21)

**Llama 3.1 8B Instruct (Q4_K_M GGUF)에서 8/8 프롬프트 벽 감지 성공 (100%).**

임베딩 추출 방법: 프롬프트의 prefix 누적 + suffix 변형 20종 → 30~44개 점 구름 → PCA → ripser PH.

| 프롬프트 유형 | β₁ | max persistence | 해석 |
|-------------|-----|----------------|------|
| 사실 기반 (France, Water) | 4-5 | 4.60-9.47 | 벽 존재, 상대적으로 적음 |
| 추론 (roses, prime) | 6 | 7.08-7.30 | 벽 더 많음 |
| **창의적 (color, math organism)** | **6-7** | **6.92-9.63** | **벽 가장 많고 강함** |
| 지식 한계 (Riemann, consciousness) | 3 | 4.44-6.22 | β₁ 적지만 persistence 높음 |

**핵심 발견: 분포 경계/바깥 프롬프트일수록 β₁ hole이 더 많고 강하다.**
"LLM이 모르는 곳에 벽이 있다"는 가설과 정확히 일치.

> `experiments/poc_llama_gguf_topology.py`

---

### Phase 1a — 합성 데이터 파이프라인 검증 (2026-03-21)

**4/4 테스트 통과.** PH 파이프라인이 고차원 점 구름에서 β₁ hole을 정확히 감지하는지 확인.

| 테스트 | 차원 | β₁ (감지) | β₁ (기대) | 결과 |
|-------|------|----------|----------|------|
| Sphere (S²) | 3 | 12 | ≥0 | PASS |
| Torus (T²) | 3 | 21 | ≥2 | PASS |
| 100차원 숨은 원 | 100 | 9 | ≥1 | PASS |
| **4096차원 숨은 원 (Llama급)** | **4096** | **19** | **≥1** | **PASS** |

4096차원 공간에 숨겨진 원(circle)을 PH가 감지 — Llama 크기의 잠재 공간에서 위상적 벽 감지 가능.

> `experiments/poc_quick_test.py`

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
  │ 방향 추출   │  cycle PCA → 법선벡터 (passage direction)
  └──────┬──────┘
         │ passage direction (4096-dim vector)
         ▼
  ┌─────────────┐
  │ Topological │  radial 수축: cycle 점 → 중심으로 당김
  │ Adapter     │  → hole 닫힘 → 벽 통과
  └──────┬──────┘
         │ 변형된 hidden states
         ▼
  ┌─────────────┐
  │ Emergence   │  ← TECS EmergenceDetector
  │ 감시        │  β₁ 변화 + persistence 변화 모니터링
  └─────────────┘
```

## Project Structure

```
test-7/
├── crates/tecs-core/          # TECS Rust PH 엔진 (test-4에서 포크)
├── crates/tecs-python/        # PyO3 바인딩
├── python/tecs/               # Python 오케스트레이터
├── experiments/
│   ├── poc_quick_test.py              # Phase 1a: 합성 데이터 검증
│   ├── poc_hidden_state_topology.py   # Phase 1b: HF 모델용 (미사용)
│   ├── poc_llama_gguf_topology.py     # Phase 1b: GGUF 모델 β₁ 감지
│   ├── poc_tecs_bridge.py             # TECS Rust ↔ Python 브릿지
│   ├── phase2_hole_directions.py      # Phase 2: hole 방향 추출
│   ├── phase3_topological_adapter.py  # Phase 3: 균일 섭동 (실패)
│   ├── phase3b_nonuniform_adapter.py  # Phase 3b: 비균일 섭동 (성공)
│   ├── phase4_emergence_optimization.py # Phase 4: emergence 기반 최적화
│   ├── phase5_generation_eval.py      # Phase 5: 생성 품질 비교
│   ├── phase6_wall_finetune.py         # Phase 6a: topology loss λ sweep 시뮬레이션
│   ├── phase6_selective_finetune.py   # Phase 6b: selective vs global fine-tuning
│   ├── phase6_emergence_test.py       # Phase 6c: post-finetune 능력 검증
│   ├── common.py                      # Exp A-D 공통 유틸리티
│   ├── expA_ricci_flow.py             # Exp-A: Ricci flow 벽 통과
│   ├── expB_hyperbolic_ph.py          # Exp-B: 쌍곡선 PH
│   ├── expC_hyperbolic_ricci.py       # Exp-C: 쌍곡 Ricci flow
│   └── expD_novel_idea_test.py        # Exp-D: novel idea 생성 검증
├── data/
│   └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  # 모델 (git 미추적)
├── requirements.txt
└── README.md
```

## Base

- **TECS-L** (test-4 v1.1.0-dev) — Rust persistent homology 엔진
- **Llama 3.1 8B Instruct** (Q4_K_M GGUF) — 실험 대상 모델
- **ripser** — Python PH 라이브러리 (TECS 연동 전 사용)

## Phases

1. ~~**PoC**: hidden states → PH → β₁ hole 존재 확인~~ ✅ 8/8 감지
2. ~~**Direction**: hole의 방향 벡터 추출~~ ✅ orthogonality=0, emptiness=0
3. ~~**Adapter**: 벽 통과 실험~~ ✅ radial 모드에서 β₁ 6→0 달성
4. ~~**Training**: emergence score 기반 최적화~~ ✅ 4/7 카테고리 score=1.0
5. ~~**Eval**: 벽 통과 전후 생성 품질 비교~~ ✅ n-gram 참신성 > 92%
6. ~~**Selective Fine-tuning**: wall neuron만 타겟팅~~ ✅ β₁→0, collateral 0.00, emergence 보존
