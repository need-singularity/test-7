# test-7: Topological Wall Passage for LLMs

Llama 8B의 잠재 공간에서 위상적 "벽"(β₁ hole)을 감지하고,
그 벽을 통과하는 adapter를 구현하는 실험 프로젝트.

---

## Experiment Results

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
│   └── phase3b_nonuniform_adapter.py  # Phase 3b: 비균일 섭동 (성공)
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
3. ~~**Adapter**: 벽 통과 실험~~ ✅ radial 모드에서 β₁ 6→3 달성
4. **Training**: emergence score 기반 학습 루프
5. **Eval**: 벽 통과 전후 생성 품질 비교
