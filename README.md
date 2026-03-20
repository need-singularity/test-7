# test-7: Topological Wall Passage for LLMs

Llama 8B의 잠재 공간에서 위상적 "벽"(β₁ hole)을 감지하고,
그 벽을 통과하는 adapter를 구현하는 실험 프로젝트.

## Core Idea

- TECS의 persistent homology로 hidden states의 위상 구조 분석
- β₁ hole = 표현 공간의 "벽" (분포 경계)
- Topological Adapter = 벽을 우회하는 새 차원 주입

## Base

- TECS-L (test-4 v1.1.0-dev) Rust 위상 엔진
- Llama 3.1 8B (또는 호환 모델)

## Phases

1. **PoC**: hidden states → PH → β₁ hole 존재 확인
2. **Direction**: hole의 방향 벡터 추출
3. **Adapter**: TopologicalAdapter 구현
4. **Training**: emergence score 기반 학습
5. **Eval**: 벽 통과 전후 비교
