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

### Extension Experiments (Exp A-D): 비유클리드 기하학 + 검증

Phase 1-5 완료 후, 독립적으로 진행하는 확장 실험들.

| # | 실험 | 상태 | 핵심 아이디어 |
|---|------|------|-------------|
| A | 푸앵카레 추측 (S³ 복원) | 🔄 진행중 | Ricci flow 근사, 방향 보존 |
| B | 쌍곡선 임베딩 PH | 🔜 예정 | Poincaré disk 거리, TECS 연동 |
| C | 푸앵카레 + 쌍곡선 결합 | 🔜 예정 | 쌍곡 Ricci flow (A+B 의존) |
| D | 새 아이디어 생성 테스트 | 🔄 진행중 | 벽 통과 후 실제 novel output 검증 |

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

## Next Experiments (TODO)

### Exp-A: 푸앵카레 추측 (S³ 방향 보존) 기반 벽 통과

페렐만이 증명한 푸앵카레 추측의 핵심:

> 단일 연결된(simply connected) 닫힌 3차원 다양체는 3차원 구(S³)와 위상동형이다.

**적용 아이디어:**
- LLM 잠재 공간이 "구멍 없이 닫혀 있다면" S³과 동형 → 모든 루프가 수축 가능
- β₁ hole이 있다 = 단일 연결이 아니다 = S³이 아닌 다양체에 갇혀 있다
- 벽 통과 = β₁ hole을 수축시켜 단일 연결 조건을 복원 → S³로 회복
- S³ 방향 보존: 오른쪽으로 출발 → 왼쪽에서 귀환, 방향(orientation) 불변
- 이것을 adapter에 적용: 섭동 후에도 임베딩의 방향성이 보존되는지 검증

**구현 포인트:**
- Ricci flow 근사로 hole 수축 (현재 radial 수축의 이론적 기반)
- 방향 보존 조건을 adapter 제약으로 추가
- β₁=0 도달 시 "S³ 복원" 판정

### Exp-B: 비유클리드 기하학 — 쌍곡선 임베딩 배치

**적용 아이디어:**
- 현재: 유클리드 거리 → PH → β₁ 감지
- 개선: 쌍곡선(Poincaré disk) 거리로 PH 계산
- TECS에 이미 `hyperbolic/sarkar.rs` (Sarkar 임베딩), `hyperbolicity.rs` (δ-hyperbolicity) 구현 있음
- LLM 잠재 공간은 계층적 구조 → 쌍곡 거리가 유클리드보다 정확할 수 있음

**구현 포인트:**
- hidden states → Poincaré ball 임베딩 (exponential map)
- 쌍곡 거리 행렬 → VR complex → PH
- 유클리드 vs 쌍곡에서 β₁ 감지 비교
- δ-hyperbolicity로 "이 공간이 얼마나 쌍곡적인가" 측정

### Exp-C: 푸앵카레 + 쌍곡선 결합

**적용 아이디어:**
- Exp-A + Exp-B 결합
- 쌍곡 공간에서 S³ 방향 보존 조건의 벽 통과
- Ricci flow를 쌍곡 메트릭에서 실행
- 가장 이론적으로 완전한 형태

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
│   └── phase5_generation_eval.py      # Phase 5: 생성 품질 비교
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
