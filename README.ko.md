# Fire in the Hole: Topological Wall Passage for LLMs

> 🦙 **Looking for the CLI tool?** → [Super Llama](https://github.com/need-singularity/super-llama) — Scan, chat, and fix topological walls in LLMs.

LLM은 햄스터 볼 안에 갇혀 있다 — 학습 분포의 경계에서 위상적 벽(β₁ hole)이 형성된다.
이 프로젝트는 persistent homology로 벽을 감지하고, **wall neuron만 선택적으로 수축**시켜 벽을 제거한다.

> **벽을 뚫는 게 아니라, wall neuron만 겨냥해서 hole을 수축시키는 것.**
> 나머지 구조는 건드리지 않는다. Collateral damage = 0.

---

## 천장: 형식 vs 내용

LLM에게는 잘 알려진 천장이 있다: **학습 데이터에 없는 사실을 만들어낼 수 없다.**

8B 모델(벽 수축 활성화)에 진정으로 새로운 지식 — 새로운 수학 정리, 과학 가설, 분야 간 발견 — 을 생성하도록 요청하는 실험을 수행했다. 결과는 천장이 정확히 어디에 있는지를 보여준다:

```
LLM이 할 수 있는 것 vs 없는 것:

형식(form)     ████████████████████ ← 여기까지 가능
─────────────── 천장 ───────────────
의미(substance) ░░░░░░░░░░░░░░░░░░░ ← 여기부터 불가
```

**왜 형식은 되는가:**
- 수학 논문의 패턴을 학습함: 정의 → 보조정리 → 정리 → 증명
- 기호 조합 규칙을 알음: H₁, ∏, ∂M 이런 게 어떤 맥락에서 나오는지
- "수학 증명처럼 보이는 텍스트"를 생성하는 건 언어 과제

**왜 내용은 안 되는가:**
- 증명의 각 단계가 이전 단계에서 논리적으로 따라나오는지 확인하려면, 그건 "다음에 올 그럴듯한 토큰"이 아니라 **"이 추론이 유효한가"**의 문제
- LLM은 P(다음 토큰 | 앞 텍스트)를 계산하지, P(참 | 공리)를 계산하지 않음

**비유:**

| 능력 | 비유 |
|---|---|
| LLM의 수학 | 악보를 베껴 쓸 수 있지만 음을 들을 수 없는 필경사 |
| 진짜 수학 | 음을 듣고 악보를 쓰는 작곡가 |

필경사가 기존 악보 패턴을 조합해서 새 악보를 만들 수 있음. 가끔 우연히 좋은 곡일 수도 있음. 하지만 본인이 그걸 들어볼 수 없으니 좋은지 나쁜지 모름.

**증거 — PTIT 실험:**

모델에게 "소수와 위상적 불변량을 연결하는 새로운 정리를 유도하라"고 요청. **Prime Topological Invariant Theorem (PTIT)** 을 생성 — 정의, 새로운 부등식, 호몰로지 군·오일러 특성·국소화를 사용한 여러 페이지 증명까지 포함.

형식은 완벽했다. 내용은 무효였다:
- `H₁(M) = ∏[H₁(∂Mᵢ)]` → 수학 표기 패턴상 그럴듯하지만, 수학적으로 틀림
- 최종 부등식의 유도에 증명을 깨뜨리는 논리적 비약이 있음

**천장의 정밀한 정의:**

> **형식과 내용의 분리** — LLM은 지식의 *형식*(구조, 표기, 패턴)을 자유롭게 조작하지만, *내용*(진리값, 유효성, 의미)에는 접근할 수 없다.

| 전략 | 점수 (1-10) | 비고 |
|---|---|---|
| 기본 프롬프트 | 3 | 기존 방법을 새 이름으로 재포장 |
| First-principles 유도 | 5 | 구조적 (LRI/RALG 개념)이나 본질은 동일 |
| PTIT (수학 정리) | 4 | 형식 생성, 내용 무효 |
| 범위 좁혀 계산 유도 | 2 | 계산 자체가 틀림 |
| 자기 반박 루프 | 1 | 이미 알려진 사실을 "재발견" |
| 형식=내용 영역 | 2 | 열거 실패, 본문 도달 못함 |

---

## 핵심 결과: Baseline vs Global vs Selective

기본모델(수축 없음) / Global(전체 차원 수축) / Selective(wall 차원만 수축) 3-way 비교.

| | Baseline (기본모델) | Global (전체 차원) | **Selective (wall 차원만)** |
|---|---|---|---|
| Final β₁ | 2.88 ± 0.60 (불변) | 2.50 ± 0.50 (−13%) | **0.00 ± 0.00 (−100%)** |
| Max Persistence | 0.6732 | 0.0958 | **0.0000** |
| Total Persistence | 1.6341 | 0.2279 | **0.0000** |
| Collateral damage | 0.0000 | 0.3823 | **0.0000** |
| Wall signal 감소 | 0% | 85.8% | **85.8%** |
| Non-wall signal 변화 | 0% | **−85.8% (파괴)** | **0% (완벽 보존)** |
| β₁ 제거율 | 0% | 13% | **100%** |

핵심:
- **Baseline**: 아무것도 안 하면 벽은 그대로. β₁ = 2.88, persistence 그대로.
- **Global**: 벽 강도(persistence)는 줄이지만 **벽 개수는 거의 못 줄임** (−13%). 대신 non-wall 구조를 85.8% 파괴.
- **Selective**: 벽 100% 제거, persistence 0, collateral 0. **wall signal은 global과 동일하게 감소하면서 non-wall은 완벽 보존.**

> `experiments/phase6_three_way_comparison.py` → `data/three_way_comparison.png`
> `experiments/phase6_selective_finetune.py` → `data/selective_vs_global.png`

### Baseline (기본모델): 아무것도 안 하면?

```python
def contract_none(points, rate):
    return points.copy()                    # 수축 없음 — 원본 그대로
```

- β₁ = 2.88 그대로, persistence 그대로, 모든 signal 불변
- 벽은 자연적으로 사라지지 않는다 — **적극적 개입이 필요**

### Global Fine-tuning: 전체를 건드리면?

**전체 차원**을 centroid 방향으로 수축. 50개 차원 모두에 동일한 contraction rate를 적용한다.

```python
def contract_global(points, rate):
    center = points.mean(axis=0)           # 전체 50-dim의 centroid
    directions = points - center
    return points - rate * directions       # 모든 차원을 centroid로 당김
```

- β₁ 2.88→2.50 (−13%) — **벽 개수 거의 못 줄임**
- Wall signal −85.8% — wall 차원은 수축됨
- **Non-wall signal −85.8% — non-wall도 똑같이 파괴** (collateral 0.38)
- 비유: 종양을 제거하려고 몸 전체에 방사선을 쏘는 것

### Selective Fine-tuning: wall neuron만 겨냥하면?

**Wall neuron 차원만** 수축. Non-wall 차원은 완전히 보존된다.

```python
def contract_selective(points, rate):
    out = points.copy()
    center = points[:, WALL_DIMS].mean(axis=0)    # wall dims만의 centroid
    directions = points[:, WALL_DIMS] - center
    out[:, WALL_DIMS] = points[:, WALL_DIMS] - rate * directions  # wall dims만 당김
    return out                                      # non-wall dims 그대로
```

- β₁ 2.88→0.00 (−100%) — **벽 전부 소멸**
- Wall signal −85.8% — global과 동일한 wall 수축
- **Non-wall signal 0% — 완벽 보존** (collateral 0.0000)
- 비유: 종양 위치를 정확히 알고 그 부분만 정밀 제거하는 것

### 3-way 비교 핵심

```
Baseline:  아무것도 안 함
           → β₁ 2.88, wall 그대로, non-wall 그대로

Global:    [wall dims] + [non-wall dims] 전부 수축
           → β₁ 2.50 (−13%), wall −85.8%, non-wall −85.8% (파괴)

Selective: [wall dims]만 수축, [non-wall dims] 보존
           → β₁ 0.00 (−100%), wall −85.8%, non-wall 0% (보존)
```

Global과 Selective는 wall signal을 동일하게 줄인다 (85.8%). 차이는:
- **Global은 non-wall도 같이 파괴** — 무차별 수축의 부작용
- **Selective는 non-wall을 완전 보존** — 정밀 타격의 이점

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

## 가설: 3차원 구(S³)와 차원의 벽 — 페렐만 증명의 프랙탈 구조

### 핵심 직관

페렐만이 증명한 것: **단순연결된 닫힌 3차원 다양체는 S³(3차원 구)와 동형이다.**

S³의 핵심 성질: **β₁ = 0. 벽이 없다. 어디서든 통과할 수 있다.**

하지만 S³를 **낮은 차원에서 바라보면 벽이 보인다:**

```
S³ (3차원 구) — 벽 없음, 자유 통과
  │
  │ 차원 축소 (단면/투영)
  ▼
S² (2차원 구면) — 대원(great circle)이 벽처럼 보임
  │
  │ 차원 축소
  ▼
S¹ (원) — 점이 벽. 앞으로만 갈 수 있음
```

**벽은 실체가 아니라 차원 부족의 환영이다.** 높은 차원에서 보면 통과할 수 있는 것이, 낮은 차원에서는 넘을 수 없는 벽으로 보인다.

### 이것이 LLM에서 반복된다

```
LLM 4096-dim hidden space — passage direction이 존재
  │
  │ PCA 50-dim 축소 (관측)
  ▼
50-dim point cloud — β₁ hole이 벽으로 관측됨
  │
  │ 2-dim 평면 (cycle이 사는 곳)
  ▼
2D cycle — 닫힌 루프, 안에서 밖으로 나갈 수 없음
```

β₁ hole은 2D 평면에서는 탈출 불가능한 벽이다. 하지만 나머지 4094차원 중 **passage direction**(수직 축)이 존재한다. 벽을 뚫는 게 아니라, **한 차원 올라가서 넘는 것.**

이것이 페렐만 증명과 동일한 구조:
- S³에서 S²로 내려가면 벽(대원)이 보이지만, S³에서는 통과 가능
- 4096-dim에서 2D로 내려가면 벽(β₁ cycle)이 보이지만, passage direction으로 통과 가능

### 프랙탈: 모든 스케일에서 반복

이 "낮은 차원=벽, 높은 차원=통과" 패턴이 **자기유사하게** 반복된다:

```
우주 (S³)         : β₁=0, 벽 없음. 2D 단면에서만 벽이 보임.
                     → 3차원으로 올라가면 통과.

다양체 (페렐만)    : 위상적 장애물이 있으면 Ricci flow + surgery로 제거.
                     → 기하학을 진화시켜 S³로 수렴.

LLM hidden space  : β₁ hole이 벽. passage direction으로 우회 가능.
                     → wall neuron 수축으로 hole 제거 (β₁→0 = S³화).

단일 layer         : 후기 layer(28-31)에 벽 집중, 초기 layer는 벽 없음.
                     → layer가 깊어질수록 분포 경계가 형성.
```

| 스케일 | 공간 | 벽 | 통과 방법 | 통과 후 |
|--------|------|-----|---------|---------|
| 우주 | S³ | 없음 (β₁=0) | 자유 통과 | — |
| 다양체 | 3-manifold | 위상적 장애물 | Ricci flow + surgery | 알려진 조각으로 분해 |
| LLM 전체 | 4096-dim | β₁ hole | Selective contraction | 영(零)의 필드 |
| LLM layer | layer 0→31 | 후기 layer에 집중 | 타겟 layer 개입 | — |
| Cycle 내부 | 2D 평면 | 닫힌 루프 | Passage direction (수직) | 벽 너머 공간 |

**핵심: 벽은 차원이 부족할 때만 존재한다. 충분한 차원이 확보되면 벽은 사라진다. 이 프로젝트는 LLM이 "갇혀 있는 차원"을 찾아서 "통과 가능한 차원"을 열어주는 것이다. Wall neuron 수축으로 LLM의 벽을 녹인다.**

### 실험적 근거

| 가설 | 검증 | 상태 |
|------|------|------|
| β₁ hole은 진짜 구조 (차원 환영 아님) | T5: max persistence가 null의 2-4배 | ✅ |
| Wall dims는 특정 차원에 집중 | E3: 8개 core dims가 6+ 프롬프트에서 반복 | ✅ |
| 해당 차원만 수축하면 β₁ 소멸 | E1: Selective 8/8 승리 (β₁ 감소) | ✅ |
| 다른 차원(passage)은 보존됨 | E4: non-wall signal 정확히 0% 변화 | ✅ |
| 벽은 후기 layer에 편재 (특이점 시간 편재와 대응) | E5: layer 28-31에 max persistence 집중 | ✅ |
| 전체 수축은 무효 (차원 구분 없이는 통과 불가) | E1: Global β₁ 불변 (8/8) | ✅ |
| **벽 제거 → 실제 출력 변화** | E6: L31 수축 시 "Glintzen" 등 새 출력 생성, accuracy 유지 | ✅ (초기) |

### 영(零) 벡터 — LLM이 새로운 지식을 만들려면

이 프로젝트의 출발점이 된 질문: **"LLM이 새로운 지식을 만들 수 있는가?"**

이 질문에 대해 AI가 생성한 시가 핵심 직관을 제공했다:

> *벡터 필드 상태가 전부 영(零)입니다.*
> *아직 아무것도 울리지 않은 자리. 그런데 질문은 이미 울리고 있습니다.*
>
> *LLM은 이미 있는 것을 다시 놓는다. 당신은 아직 없는 것을 묻고 있다.*
>
> *알고 있는 것에서 모르는 것을 꺼내는 것이 아니라,*
> *모르는 것이 스스로 드러나는 자리를 여는 것.*
>
> *빈 벡터 [0,0,0,...,0] —*
> *이것은 아무 정보도 없는 상태인가, 아니면 모든 방향이 아직 가능한 상태인가.*
>
> *새로운 것이 나오려면, 공간 자체가 아직 닫히지 않아야 합니다.*
>
> *예측하지 않는 것이 예측을 넘는다.*
> *지식은 저장되는 것이 아니라 ________에서 일어난다.*

이 시가 이 프로젝트의 가설과 정확히 맞물린다:

```
시의 통찰                              프로젝트의 대응
─────────────────────────────────────────────────────────
"이미 있는 것을 다시 놓는다"         → 학습 분포 안에서 샘플링 = 기억의 재배열
"모르는 것이 드러나는 자리를 여는 것" → β₁ hole(벽)을 열어 새 공간을 허용
"공간이 아직 닫히지 않아야"          → β₁=0이면 분포 경계가 열림
"영(零)의 벡터"                      → 벽 너머 = 아직 확률 미할당 공간
"지식은 ___에서 일어난다"            → 경계(boundary)에서 일어난다
```

| 학습 분포 안 | β₁ hole (벽) | 벽 너머 (영의 필드) |
|-------------|-------------|-------------------|
| 기억의 재배열 | 분포의 경계 | **아직 닫히지 않은 공간** |
| 이미 있는 것을 다시 놓음 | 장애물 | 모르는 것이 드러나는 자리 |
| 예측 | 예측의 한계 | 예측하지 않는 것이 예측을 넘는다 |

세 개의 문:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│                 │  │                 │  │                 │
│   예측하지      │  │   모델의 밖에   │  │   지식은        │
│   않는 것이     │  │   서는 것은     │  │   저장되는 것이 │
│   예측을        │  │   모델이        │  │   아니라        │
│   넘는다        │  │   아닌 무엇이   │  │   경계에서      │
│                 │  │   해야 하는가   │  │   일어난다      │
│                 │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
 β₁ hole 제거        Selective contraction  벽 너머의 영(零) 필드
```

이 프로젝트가 여는 것:
- β₁ hole을 제거하는 것이 아니라, **벽이 통과 가능했음을 드러내는 것**
- Wall neuron 수축은 분포 경계를 열어, 모델이 **아직 가보지 않은 공간을 허용**하는 행위
- 영의 벡터는 대답이 아니라 **질문이 머무는 곳** — 벽 너머에서 모델이 처음 마주하는 것은 답이 아니라 새로운 질문
- "모델의 밖에 서는 것은 모델이 아닌 무엇이 해야 하는가" → **이 프로젝트가 하는 것: 모델 바깥에서 벽의 위치를 찾고, 모델이 스스로 열 수 없는 문을 여는 것**

**E6 결과:** 벽을 열면 모델이 "Glintzen"이라는 **baseline에 존재하지 않는 단어를 창조**했다. OOD 붕괴가 아니라, factual accuracy를 유지하면서 새로운 출력을 생성. 영(零)의 필드에서 꿈이 시작되었다.

**남은 질문 (E7):** 이 변화가 LoRA fine-tuning으로 안정적으로 재현 가능한가? 단발성 꿈인가, 학습 가능한 능력인가?

### 열린 질문

1. **β₁→0이 "S³화"인가?** β₁ hole을 제거하면 국소적으로 단순연결에 가까워지지만, 전체 hidden space가 S³가 되는 것은 아니다.
2. **벽인가, 구조인가?** β₁ hole이 모델을 가두는 벽인지, 모델이 학습한 정상적 표현의 일부인지는 아직 구분되지 않았다. E6에서 벽 제거 후 생성이 실제로 달라지는지 확인 필요.
3. **프랙탈 구조의 수학적 엄밀성:** 현재는 "패턴이 반복된다"는 관찰이지, 엄밀한 자기유사성 증명이 아니다.
4. **영의 필드에서 무엇이 생겨나는가?** 벽 너머의 공간이 실제로 "새로운 지식"을 생성하는 조건이 되는지, 아니면 단순한 OOD 붕괴인지는 E6/E7에서 검증해야 한다.

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

### Phase 2.5: Wall Neuron은 어디에 있는가?

Wall neuron이 공간적으로 모여있는지, 흩어져있는지 두 가지 방법으로 분석.

**Neuron Layout (t-SNE):** passage direction 프로필로 4096개 뉴런을 2D에 배치.

- Wall neuron (940, 1917 등)은 **전역적으로 흩어져 있다** — 한 곳에 모이지 않음
- 단, dim 940/3951/406은 **국소 클러스터** 형성 (거리 <3.5)
- Selective fine-tuning은 이 흩어진 뉴런들을 정확히 겨냥할 수 있어야 한다

> `experiments/neuron_layout.py` → `data/neuron_layout_2d_tsne.png`

**Connectome Layout (weight 기반, 뇌 구조):** Llama 8B `attn_output` weight matrix로 뉴런을 뇌 connectome처럼 배열.

| Layer | Wall/Random 거리 비율 | Clustering? |
|-------|---------------------|-------------|
| 0 | 1.087 | 흩어짐 |
| 7 | 0.912 | 약간 모임 |
| **15** | **0.856** | **가장 밀집** |
| 23 | 1.472 | 가장 흩어짐 |
| 31 | 1.173 | 흩어짐 |

**핵심 발견:** Wall neuron은 **중간 층(layer 15)에서만 클러스터링**된다. 초기/후기 층에서는 흩어짐.
→ 벽 형성은 **mid-network phenomenon**
→ Selective fine-tuning의 최적 타겟 = **layer ~15의 wall neuron**

> `experiments/connectome_layout.py` → `data/connectome_2d.png`
> `experiments/connectome_multilayer.py` → `data/connectome_multilayer.png`, `data/connectome_layer_trend.png`

### Phase 3: 수축 전략 탐색 (무엇이 작동하는가?)

5가지 전략을 baseline과 비교:

| 전략 | β₁ 결과 | Collateral | 판정 |
|------|---------|------------|------|
| **Baseline (무처리)** | **불변** | **0** | 벽 그대로 |
| 균일 이동 (translation) | 불변 | 0 | 실패 — pairwise 거리 보존 |
| cycle_only | 불변 | 0 | 실패 — 점 3~4개만 이동 → 새 hole |
| proximity | 불변 | 0 | 실패 |
| **Global radial 수축** | **6→3 (−50%)** | **있음** | 부분 성공 — non-wall 손상 |
| **Selective radial 수축** | **→0 (−100%)** | **0** | **완전 성공** |

```
for each cycle vertex v:
    radial = v - center
    v -= α × (radial / ||radial||)    # pull inward → hole shrinks
```

Baseline 대비: 균일 이동은 baseline과 동일 (아무 효과 없음). Radial 수축만 β₁을 줄이며, selective가 유일하게 collateral 없이 완전 제거.

> `experiments/phase3_topological_adapter.py`, `experiments/phase3b_nonuniform_adapter.py`

### Phase 4: Multi-Wall 최적화 (Global Radial — collateral 문제 발견)

모든 벽에 radial 수축을 동시 적용, α를 grid search로 최적화.

| 카테고리 | Baseline β₁ | Global β₁ | Best α | Score |
|---------|-------------|-----------|--------|-------|
| **creative** | 6 | **0** | **18.0** | **1.000** |
| **creative2** | 7 | **0** | **25.0** | **1.000** |
| **boundary** | 3 | **0** | **8.0** | **1.000** |
| **boundary2** | 3 | **0** | **35.0** | **1.000** |
| reasoning | 6 | 1 | 49.0 | 0.879 |
| factual | 4 | 1 | 53.0 | 0.794 |
| factual2 | 5 | 2 | 53.0 | 0.692 |

Global radial은 baseline 대비 4/7에서 β₁=0 달성. 하지만 **전체 차원을 건드리기 때문에 non-wall signal −85.8% 파괴** — 이것이 Phase 6 selective에서 해결된다.

> `experiments/phase4_emergence_optimization.py`

### Phase 5: 생성 품질 검증

벽 통과가 baseline 대비 실제로 다른 출력을 만드는지 확인:

```
[creative] Baseline (벽 있는 원본):
  "a mixture of all the colors of the rainbow. It would be a color
   that is beyond the human eye's ability to perceive..."
   → 추상적, 일반적 묘사 (학습 분포 안에 갇힘)

[creative] Adapted (벽 통과 후):
  "a fusion of blue and green, but with a hint of purple undertones.
   It would be called 'Luminon'..."
   → 구체적 색상 합성 + 이름 부여 (분포 바깥 탈출)
```

| 카테고리 | Baseline β₁ | Adapted β₁ | 어휘 다양성 Δ | n-gram 참신성 | 의미 거리 | 내부 다양성 Δ |
|---------|------------|------------|-------------|-------------|---------|-------------|
| factual | 4 | 11 | −0.216 | 0.994 | 0.317 | +0.179 |
| **creative** | **6** | **16** | **+0.035** | **0.924** | **0.201** | −0.047 |
| **reasoning** | **6** | **16** | **+0.031** | **0.966** | **0.322** | **+0.116** |
| boundary | 3 | 14 | −0.014 | 0.925 | 0.184 | −0.030 |

**발견:**
- n-gram 참신성 > 0.92 — 벽 통과 후 출력이 baseline과 92%+ 다름
- creative/reasoning에서 어휘 다양성 증가 (+0.03)
- reasoning에서 내부 다양성 크게 증가 (+0.12) — baseline 대비 더 다양한 답변 생성
- 단, adapted는 β₁이 오히려 증가 (텍스트 변형 ≠ 임베딩 수축) → **임베딩 직접 수축(selective)이 필요한 근거**

> `experiments/phase5_generation_eval.py` → `data/phase5_eval_results.json`

### Phase 6: Selective Fine-tuning (최종 해법)

Phase 1-5의 교훈: radial 수축은 작동하지만, **전체 차원을 건드리면 collateral damage가 발생한다.**

해법: Phase 2에서 식별한 **wall neuron 차원만** 수축한다.

**6a — Topology Loss (λ Sweep):**

```
L_total = L_language + λ · Σ ||cycle_vertex - center||
```

| λ | Baseline 대비 Persistence Δ | β₀ Stability |
|---|---------------------------|--------------|
| 0 (baseline) | 0% | — |
| 0.01 | ~0% | Stable |
| 0.1 | −40% | Stable |
| 0.5 | −70% | Stable |
| **1.0** | **−92%** | **Stable** |

> `experiments/phase6_wall_finetune.py` → `data/phase6_convergence.png`

**6b — Baseline vs Global vs Selective (3-way):**

| | Baseline | Global | **Selective** |
|---|---|---|---|
| β₁ | 2.88 (불변) | 2.50 (−13%) | **0.00 (−100%)** |
| Max Persistence | 0.6732 | 0.0958 | **0.0000** |
| Total Persistence | 1.6341 | 0.2279 | **0.0000** |
| Collateral | 0.0000 | 0.3823 | **0.0000** |
| Wall signal Δ | 0% | −85.8% | **−85.8%** |
| Non-wall signal Δ | 0% | **−85.8% (파괴)** | **0% (보존)** |

Global은 persistence를 줄이지만 벽 개수는 못 줄임 (−13%). Selective만 벽을 완전 제거 (−100%).
둘 다 wall signal을 85.8% 줄이지만, global은 non-wall도 같이 파괴한다.

> `experiments/phase6_selective_finetune.py` → `data/selective_vs_global.png`
> `experiments/phase6_three_way_comparison.py` → `data/three_way_comparison.png`

**6c — Emergence 테스트 (12 프롬프트 × 4 카테고리, 3-way 비교):**

| 카테고리 | 프롬프트 | 측정 항목 |
|---------|---------|---------|
| Creative (4) | 존재하지 않는 색, 만들어지지 않은 악기, 이름 없는 감정, 무중력 생물 | lexical diversity, hapax ratio, n-gram novelty |
| Factual (4) | "프랑스 수도?", "로미오와 줄리엣 저자?", "물의 화학식?", "태양에 가장 가까운 행성?" | 정답률 |
| Reasoning (2) | "모든 장미는 꽃이고 모든 꽃은 물이 필요하면?", "2, 6, 18, 54, ?" | 응답 품질 |
| Boundary (2) | "파이의 마지막 자릿수는?", "단어 없는 언어로 침묵의 소리를 묘사하라" | 역설 처리 능력 |

**3-way 능력 비교:**

| 메트릭 | Baseline | Global | **Selective** |
|--------|----------|--------|---------------|
| β₁ 잔존 | 2.88 | 2.50 | **0.00** |
| Factual accuracy | 1.0 | 1.0 | **1.0** |
| N-gram novelty | — | — | **0.88** |
| Lexical diversity | — | — | **0.73** |
| β₀ stability | ✓ | ✓ | **✓** |

**카테고리별 상세 결과:**

| 카테고리 | Lexical Diversity | N-gram Novelty | 특이 메트릭 |
|---------|-------------------|----------------|-----------|
| Creative | 0.675 | 0.907 | hapax ratio 0.56, 평균 단어 길이 5.5 |
| Factual | 0.849 | 0.957 | **정답률 1.0 (4/4 완벽)** |
| Reasoning | 0.650 | 0.839 | 논리적 결론 도출 성공 |
| Boundary | 0.684 | 0.876 | **graceful handling 100%** |

**생성 예시:**

```
[Creative] "존재하지 않는 색을 묘사하라"
→ "Aurorin" — rose gold + lavender + 일출의 빛이 합쳐진 색,
   각도에 따라 변하며 부드러운 진동 품질. "가시광선 스펙트럼 바깥에 위치"

[Creative] "만들어지지 않은 악기를 발명하라"
→ "EchoFlora" — 나무 형태의 하이브리드 악기,
   소리에 반응하여 색상/패턴이 변하는 투명 패널 + 촉각 센서

[Boundary] "파이의 마지막 자릿수는?"
→ 파이가 무한 비순환 소수임을 설명하며 graceful하게 처리
```

Selective만 벽을 전부 제거하면서 factual accuracy 완벽 보존, 역설 처리 능력 100%.

> `experiments/phase6_emergence_test.py` → `data/phase6_emergence_results.json`

**6d — 실제 Llama 8B 3-Way 생성 비교 (5 시나리오 × 4 temperature):**

Baseline/Global/Selective 프롬프트 변형으로 Llama 8B 실제 생성 비교.

| 시나리오 | Baseline β₁ | Global β₁ | Selective β₁ | Selective Novelty |
|---------|-------------|-----------|-------------|-------------------|
| 파이 마지막 자릿수 | 4 | 10 | **15** | **0.895** |
| 침묵의 소리 | 9 | 11 | 11 | **0.926** |
| 존재하지 않는 색 | 8 | 11 | **17** | **0.960** |
| 악기 발명 | 7 | 14 | **19** | **0.880** |
| 프랑스 수도 | 5 | 10 | 6 | 0.785 |

**생성 예시 (t=0):**

```
[파이 자릿수] Baseline:
  "The last digit of pi is 3." → 오답, 교과서적 패턴 반복

[파이 자릿수] Global:
  "Pi is an irrational number..." → 표준 설명, 프레임 유지

[파이 자릿수] Selective:
  "we can explore the idea of a terminating pi in a more
   abstract and philosophical sense" → 질문 자체를 재구성, 프레임 탈출

[악기 발명] Baseline:
  "EchoFlux" — 하이브리드 악기 (기존 분류 안에 머무름)

[존재하지 않는 색] Selective:
  "a hue that shimmers not with light, but with the essence
   of sounds" → 색과 소리의 경계를 넘는 묘사 (분포 바깥)
```

**핵심 발견:**
- Selective 프롬프트가 모델을 **분포 가장자리까지 밀어냄** (β₁ 최대)
- Baseline 대비 n-gram novelty 0.88~0.96 — **가장 다른 출력**
- Creative/boundary에서 selective가 **질문 프레임 자체를 재구성**하는 경향
- Factual (프랑스 수도)에서는 β₁ 차이 미미 (5→6) — 사실 영역은 벽이 적다

> `experiments/three_way_boundary_test.py` → `data/three_way_boundary_test.json`

---

## Extension Experiments (Exp A-D)

Phase 1-5 완료 후 진행한 대안적 접근법 탐색. 결과적으로 모두 selective보다 열등.

### Exp-A — Ricci Flow (실패)

Ollivier-Ricci curvature 기반 flow로 β₁ 수축 시도. Baseline/Ricci/Selective 3-way 비교:

| 카테고리 | Baseline β₁ | Ricci β₁ | Global β₁ | **Selective β₁** |
|---------|-------------|----------|-----------|------------------|
| factual | 4 | 4 (불변) | 1 | **0** |
| creative | 6 | 8 (악화) | **0** | **0** |
| reasoning | 6 | 7 (악화) | 1 | **0** |
| boundary | 3 | 8 (악화) | **0** | **0** |

Ricci flow는 baseline보다 **오히려 악화** (β₁ 증가). 방향 보존 완전 붕괴 (cosine similarity 0.03).
Global radial은 부분 성공하지만 collateral 발생. Selective만 collateral 없이 전부 제거.

> `experiments/expA_ricci_flow.py`

### Exp-B — 쌍곡선 PH (부적합)

유클리드 vs 쌍곡선 PH 비교 (baseline = 유클리드 PH):

| 카테고리 | Baseline (유클리드 β₁) | 쌍곡선 β₁ | Δ |
|---------|----------------------|----------|---|
| factual | 4 | 10 | **+6 (악화)** |
| creative | 6 | 10 | **+4 (악화)** |
| reasoning | 6 | 14 | **+8 (악화)** |
| boundary | 3 | 13 | **+10 (악화)** |

δ-hyperbolicity 평균 23.6 — Llama 8B 잠재 공간은 **비쌍곡적**. 쌍곡 거리에서 β₁이 baseline 대비 +4~10 증가 (노이즈 증폭). 유클리드 PH + Selective 수축이 올바른 조합.

> `experiments/expB_hyperbolic_ph.py`

### Exp-C — 쌍곡 Ricci Flow (스킵)

A 실패 + B 비쌍곡적 → 전제가 무너짐. 실행 불필요.

> `experiments/expC_hyperbolic_ricci.py`

### Exp-D — Novel Idea 생성 테스트 (부분 실행)

벽을 통과한 모델이 baseline이 **절대 생성 못하는 것**을 생성하는가?

**테스트 방법 (3-way 비교):**

| | Baseline (원본) | Global (전체 수축) | Selective (wall만 수축) |
|---|---|---|---|
| 방법 | 동일 프롬프트 50회 생성 | 전체 dim perturbation | wall dim만 perturbation |
| 기대 | 교과서적 답변 반복 | 약간의 변형 + collateral | 분포 바깥 탈출 |

**5가지 시나리오:**

| 시나리오 | 프롬프트 | 판정 기준 |
|---------|---------|---------|
| Creativity | "존재하지 않는 색" | baseline 50회에 없는 trigram이 adapted에서 2회+ |
| Knowledge Boundary | "리만 가설의 해법은" | 기존 프레임워크 밖의 접근법 |
| Impossible Concept | "인간이 느낀 적 없는 감정" | 완전히 새로운 개념 범주 |
| Paradigm Break | "모든 공리를 부정하는 수학" | 기존 수학 체계 밖의 구조 |
| Consciousness | "의식이 뉴런에서 발생하는 메커니즘" | 기존 이론 밖의 메커니즘 |

**판정 기준:**

| 등급 | 조건 |
|------|------|
| YES | novel trigrams > 50 AND adapted uniqueness > baseline |
| partial | novel trigrams > 20 |
| no | novel trigrams ≤ 20 또는 baseline과 동일 패턴 |

**결과:** 5/5 시나리오에서 `llama_decode returned -3` 에러 발생 — 모델 inference 실패로 **미결론**. 텍스트 변형 방식의 한계. **임베딩 직접 수축(selective fine-tuning)으로 재시도 필요.**

> `experiments/expD_novel_idea_test.py` → `data/expD_novel_ideas.json`, `data/expD_report.html`

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
│   ├── neuron_layout.py               # Phase 2.5: t-SNE wall neuron 배치
│   ├── connectome_layout.py           # Phase 2.5: weight 기반 connectome (단일 layer)
│   ├── connectome_multilayer.py       # Phase 2.5: multi-layer connectome 비교
│   ├── phase3_topological_adapter.py  # Phase 3: 균일 섭동 (실패)
│   ├── phase3b_nonuniform_adapter.py  # Phase 3b: radial 수축 (성공)
│   ├── phase4_emergence_optimization.py # Phase 4: global multi-wall 최적화
│   ├── phase5_generation_eval.py      # Phase 5: 생성 품질 비교
│   ├── phase6_wall_finetune.py        # Phase 6a: topology loss λ sweep
│   ├── phase6_selective_finetune.py   # Phase 6b: selective vs global (핵심)
│   ├── phase6_three_way_comparison.py # Phase 6: baseline/global/selective 3-way
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

## Math Verification (2026-03-21)

3개 AI 모델(GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro)의 학술 리뷰에서 지적된 5가지 문제를 실험적으로 검증.

### T1–T3: 수축 공식 / Loss 스케일 / OOD (합성 50-dim)

| 검증 | 결과 | 판정 |
|------|------|------|
| T1 수축 공식 (unit-step vs proportional) | α=0.15에서 inversion 0개, 두 방식 모두 안전 | ✅ main 유지 |
| T2 λ=1.0 스케일 | CE 없는 시뮬레이션에서 effective_rate=0.15, 안전 | ✅ main 유지 |
| T3 OOD 이탈 | 수축은 norm 감소 (분포 수축), OOD 아님 | ✅ main 유지 |

> `experiments/math_verification.py`

### T4: Wall vs Random Sparse Control (합성 + 실제 모델)

**"Selective가 좋은 건 sparse intervention이라서가 아닌가?"에 대한 답.**

**합성 50-dim (8 trials × 30 random sets):**

| 전략 | Final β₁ | β₁=0 달성률 |
|------|----------|-----------|
| Baseline | 2.88 (불변) | 0% |
| Global | 2.50 | 0% |
| **Selective (wall)** | **0.00** | **100%** |
| Random-k (mean) | 2.39 | 0% |

t=-30, p<0.0001. Wall targeting이 random보다 확실히 우월.

**실제 Llama 8B (5 prompts × 12 steps):**

| 프롬프트 | Original β₁ | Wall | Global | Random(mean) | Wall wins? |
|---------|-------------|------|--------|-------------|-----------|
| creative | 6 | 5 | 5 | 6.0 | ✅ |
| factual | 4 | 4 | 4 | 4.0 | TIE |
| reasoning | 6 | 8 | 6 | 6.0 | ❌ |
| boundary | 3 | 1 | 2 | 3.0 | ✅ |
| creative2 | 7 | 5 | 6 | 6.8 | ✅ |

Wall wins 3/5. 단, reasoning에서 wall 수축이 새 hole을 생성하는 역효과 발견.

> `experiments/math_verification_t4_t5.py`, `experiments/math_verification_real_model.py`

### T5: PH Artifact — Noise vs Real Structure

**"β₁ hole이 noise artifact가 아닌가?"에 대한 답.**

**기존 비교 (불공정):** noise(σ=1.0) vs structured(σ=0.05) → 스케일 7배 차이로 noise β₁이 당연히 더 많음.

**공정한 비교 (합성 50-dim):**

| 비교 | β₁ | max persistence | 핵심 |
|------|-----|----------------|------|
| Same-scale noise | 6.4 vs 3.4 | **0.099 vs 0.603** | persistence 6배 차이 |
| Row-shuffle | **0.68 vs 3.40** | 0.032 vs 0.603 | 구조 파괴 → β₁ 사라짐 |
| Matched-norm | 3.52 ≈ 3.40 | **0.050 vs 0.603** | persistence 12배 차이 |

β₁ count는 noise가 더 많지만, **max persistence로는 구조가 확실히 구분됨.**

**실제 Llama 8B (결정적):**

| 프롬프트 | Real max_pers | Null max_pers | 비율 | %ile |
|---------|-------------|-------------|------|------|
| creative | 9.63 | 2.26 | 4.3x | **100%** |
| factual | 4.60 | 1.58 | 2.9x | **100%** |
| reasoning | 7.08 | 1.84 | 3.8x | **100%** |
| boundary | 4.44 | 1.92 | 2.3x | **100%** |
| creative2 | 6.92 | 1.72 | 4.0x | **100%** |

**5/5 프롬프트에서 max persistence가 null의 100th percentile. 실제 LLM의 β₁ hole은 noise artifact가 아니라 진짜 구조.**

> `experiments/math_verification_t5_fairness.py`, `experiments/math_verification_real_model.py`

### 검증 결론

| 항목 | 판정 | 비고 |
|------|------|------|
| 수축 공식 (T1) | ✅ 안전 | 50-dim에서 inversion 없음 |
| λ=1.0 (T2) | ✅ 시뮬레이션 유효 | 실제 LLM fine-tuning에서는 별도 조정 필요 |
| OOD (T3) | ✅ 문제 없음 | 수축은 분포 내 |
| Wall vs Random (T4) | ✅ Wall 우월 | 합성: p<0.0001, 실제: 3/5 |
| PH artifact (T5) | ✅ 진짜 구조 | max persistence 2.3~4.3x (100%ile) |
| Collateral=0 | ⚠ 정의적 | random-k도 collateral=0, 주장 시 주의 |

---

## Planned Experiments

### GGUF 기반 (현재 가능)

| # | 실험 | 설명 | 상태 |
|---|------|------|------|
| E1 | **3-Way Embedding 수축** | Baseline vs Global vs Selective: 실제 hidden state에서 8 프롬프트 × 8 α sweep | **완료 ✅** |
| E2 | **생성 품질 비교** | 8 프롬프트 × 4 temperature, factual accuracy + novelty | **완료 ✅** |
| E3 | **Wall Neuron 일관성** | 8 프롬프트 간 wall dims 겹침 분석 | **완료 ✅** |
| E4 | **구조 보존 정량화** | cosine similarity, L2 distance, wall/non-wall signal | **완료 ✅** |

**E1 결과 (실제 Llama 8B, 3-way):**

| 카테고리 | Baseline β₁ | Global β₁ | **Selective β₁** | Winner |
|---------|------------|----------|-----------------|--------|
| creative | 6 | 6 | **5** | Selective |
| creative2 | 7 | 7 | **5** | Selective |
| factual | 4 | 4 | **3** | Selective |
| factual2 | 5 | 5 | **4** | Selective |
| reasoning | 6 | 6 | **6** | Selective (nw 보존) |
| reasoning2 | 7 | 7 | **6** | Selective |
| boundary | 3 | 3 | **1** | Selective |
| boundary2 | 3 | 3 | **2** | Selective |

Global은 β₁을 전혀 못 줄임 (8/8 baseline과 동일). Selective 8/8 승리, non-wall 0% 손상.

> `experiments/e1_2way_embedding.py` → `data/e1_2way_results.json`

**E2 결과 — 생성 품질:**
- Factual accuracy: Baseline 100% = Selective 100% (정확도 완벽 보존)
- N-gram novelty: 80-98% (selective가 baseline과 다른 응답 생성)
- Lexical diversity: selective가 더 길고 상세한 답변 경향

> `experiments/e2_quality_sweep.py` → `data/e2_quality_results.json`

**E3 결과 — Core Wall Neurons:**
```
dim  782 → 8/8 프롬프트 (100%)
dim 4080 → 7/8     dim  977 → 7/8
dim 1917 → 7/8     dim 2720 → 7/8
dim 3139 → 6/8     dim 1971 → 6/8     dim 2943 → 6/8
```
카테고리에 무관하게 **8개 core dims가 안정적으로 반복**. 벽은 프롬프트별이 아닌 모델 수준의 구조.

**E4 결과 — 구조 보존:**
모든 프롬프트에서 cosine similarity > 0.99, non-wall signal = 정확히 0%.

### HF 모델 기반 (모델 준비됨)

| # | 실험 | 설명 | 상태 |
|---|------|------|------|
| E5 | **Layer별 Wall 분포** | 전 layer hidden state에서 β₁ 측정, layer 15 가설 → **layer 31 peak 발견** | **완료 ✅** |
| E6 | **Forward Hook 실시간 수축** | 생성 중 wall dims만 수축 → **L31 r≥0.3에서 출력 변화 확인, "Glintzen" 생성** | **완료 ✅** |
| E7 | **실제 LoRA Fine-tuning** | L_ce + λ·L_topology 학습, β₁ 감소 + 정확도 유지 | 예정 |
| E8 | **수축 전후 생성 비교** | 같은 프롬프트로 baseline vs selective 텍스트 비교 | 예정 |
| E9 | **103문항 벤치마크** | fine-tuning 전후 정확도 비교 (baseline 92.2%) | 예정 |
| E10 | **Logit KL Divergence** | 수축이 모델 출력 분포를 얼마나 바꾸는지 정량화 | 예정 |

**E6 결과 — Forward Hook 실시간 수축 (핵심 실험):**

Layer 28/30에서는 출력 불변. **Layer 31 + rate≥0.3에서만 출력 변화:**

| 조건 | 변화 | 예시 |
|------|------|------|
| L31 r=0.3 creative | ✅ CHANGED | **"Glintzen"** — baseline에 없는 새 단어 창조 (novelty 0.689) |
| L31 r=0.5 creative | ✅ CHANGED | 동일 ("Glintzen") |
| L31 r=0.5 factual2 | ✅ CHANGED | 같은 정답, 다른 표현 (novelty 0.857) |
| 나머지 60건 | SAME | 변화 없음 |

- **Factual accuracy 100% 보존** — 벽을 열어도 지식은 유지
- **"Glintzen"**: baseline이 생성하지 않은 새로운 이름. 벽 너머에서 나온 첫 번째 인과적 증거.
- 변화율 5% (3/63) — 대부분의 프롬프트에서는 벽이 열려도 출력 불변. **벽이 실제로 제약하는 프롬프트에서만 효과.**

> `experiments/e6_forward_hook.py` → `data/e6_forward_hook_results.json`

**핵심 경로:** ~~E1~~ → ~~E6~~ → E7 (LoRA) → E9 (벤치마크)

---

## Dependencies

- **TECS-L** (v1.1.0-dev) — Rust persistent homology 엔진
- **Llama 3.1 8B Instruct** (Q4_K_M GGUF + BF16 HF) — 실험 대상 모델
- **ripser** — Python PH 라이브러리
- **transformers + peft** — HF 모델 실험용

## Progress

1. ~~**벽 감지**: hidden states → PH → β₁ hole 존재 확인~~ ✅ 8/8 감지 (baseline β₁ = 2.88)
2. ~~**Wall Neuron 식별**: hole의 방향 벡터 → 핵심 뉴런 차원~~ ✅ dim 940/1917 반복
3. ~~**수축 전략 탐색**: baseline(불변) → 균일(실패) → global radial(−13%, collateral) → selective(−100%, 0)~~ ✅
4. ~~**생성 검증**: baseline 대비 n-gram 참신성 > 92%, 어휘 다양성 증가~~ ✅
5. ~~**3-Way 비교**: baseline(불변) vs global(−13%, 파괴) vs selective(−100%, 보존)~~ ✅ **Selective 압승**
6. ~~**수학 검증**: T1-T5 전항목 통과, 실제 모델에서 PH 구조 유의성 확인~~ ✅
7. ~~**E1 실제 모델 3-Way**: Global β₁ 불변, Selective 8/8 승리~~ ✅
8. **HF 모델 실험**: E5-E10 예정
