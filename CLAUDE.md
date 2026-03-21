# Fire in the Hole — Project Guidelines

## Language
- 항상 한국어로 응답

## Experiments
- 실험은 백그라운드로 실행하고, 완료 시 결과를 간결하게 보고
- 긴 실험 중에는 주기적으로 진행 상황 보고 (중간 출력 체크)
- 실험 결과는 `data/` 디렉토리에 JSON으로 저장

## Models
- GGUF: `data/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- HF (BF16): `data/llama-3.1-8b-instruct/` (symlink → hamster-ball)
- 백업: `/Volumes/Ultra Touch/AI Model/llama-3.1-8b/`

## Python Environment
- 이 프로젝트에 .venv 없음. hamster-ball venv 사용: `/Users/ghost/Dev/hamster-ball/.venv/bin/python`

## Git
- main 브랜치에서 작업
- 실험 추가 시 README Planned Experiments 테이블 업데이트
