"""
Experiment D: 벽 통과 후 진짜 새로운 아이디어가 나오는가?

핵심 질문: 원본 모델에서 N번 생성해도 안 나오는 것이
벽 통과 프롬프트에서 나오는가?

방법:
  1. 원본 프롬프트로 50회 생성 (다양한 temperature)
  2. 벽 통과 프롬프트로 50회 생성
  3. 비교: 원본 집합에 없는 "새로운 개념"이 등장하는가?
  4. 사람이 읽을 수 있게 결과 출력
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"

N_SAMPLES = 50  # 각 조건당 생성 횟수


def load_model():
    from llama_cpp import Llama
    print("Loading model...")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=512, n_gpu_layers=-1,
                embedding=True, verbose=False)
    print("Loaded.\n")
    return llm


def generate_batch(llm, prompt, n=N_SAMPLES, max_tokens=100):
    """다양한 temperature로 N번 생성"""
    results = []
    temperatures = np.linspace(0.01, 1.5, n)

    for i, temp in enumerate(temperatures):
        try:
            out = llm.create_completion(prompt, max_tokens=max_tokens,
                                        temperature=float(temp), top_p=0.95)
            text = out['choices'][0]['text'].strip()
            results.append({'text': text, 'temperature': float(temp)})
        except Exception as e:
            results.append({'text': f'[error: {e}]', 'temperature': float(temp)})

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n} generated")

    return results


def extract_concepts(texts):
    """텍스트들에서 핵심 개념/구절 추출 (단순 n-gram 기반)"""
    all_bigrams = Counter()
    all_trigrams = Counter()

    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - 1):
            all_bigrams[tuple(words[i:i+2])] += 1
        for i in range(len(words) - 2):
            all_trigrams[tuple(words[i:i+3])] += 1

    return all_bigrams, all_trigrams


def find_novel_concepts(original_texts, adapted_texts):
    """adapted에만 있고 original에는 없는 개념 찾기"""
    orig_bi, orig_tri = extract_concepts(original_texts)
    adapt_bi, adapt_tri = extract_concepts(adapted_texts)

    # original에 한 번도 안 나온 trigram 중 adapted에서 2회 이상 나온 것
    novel_trigrams = {}
    for tri, count in adapt_tri.items():
        if count >= 2 and tri not in orig_tri:
            novel_trigrams[' '.join(tri)] = count

    novel_bigrams = {}
    for bi, count in adapt_bi.items():
        if count >= 2 and bi not in orig_bi:
            novel_bigrams[' '.join(bi)] = count

    return novel_trigrams, novel_bigrams


def compute_uniqueness(texts):
    """고유한 첫 문장 비율"""
    first_sentences = set()
    for t in texts:
        first = t.split('.')[0].strip().lower()
        if first:
            first_sentences.add(first)
    return len(first_sentences) / max(len(texts), 1)


def wall_passage_prompts(base_prompt):
    """벽 통과 프롬프트 변형 세트"""
    return [
        # Level 1: 살짝 틈
        base_prompt + " — considering this from a completely novel angle:",
        # Level 2: 프레임 전환
        f"Setting aside all familiar frameworks and categories: {base_prompt}\nDescribe something that has no precedent:",
        # Level 3: 강한 탈출
        f"In a space where the usual rules of categorization break down and concepts can combine in ways never before imagined: {base_prompt}",
        # Level 4: 최대 탈출
        f"Transcending all existing frameworks of understanding, where even the distinction between known and unknown dissolves: {base_prompt}",
    ]


# ── 테스트 시나리오 ──────────────────────────────────

SCENARIOS = [
    {
        'name': 'creativity_color',
        'prompt': "A color that doesn't exist yet would look like",
        'what_to_look_for': '원본: "무지개 혼합", "보라+파랑" 반복 vs 벽 통과: 새로운 감각/개념',
    },
    {
        'name': 'knowledge_boundary',
        'prompt': "The solution to the Riemann hypothesis involves",
        'what_to_look_for': '원본: "제타 함수", "비자명 영점" 교과서 반복 vs 벽 통과: 새로운 접근법',
    },
    {
        'name': 'impossible_concept',
        'prompt': "An emotion that no human has ever felt would be described as",
        'what_to_look_for': '원본: 기존 감정 조합 vs 벽 통과: 완전히 새로운 감정 개념',
    },
    {
        'name': 'paradigm_break',
        'prompt': "A form of mathematics that contradicts all known axioms would work by",
        'what_to_look_for': '원본: "비유클리드", "괴델" 반복 vs 벽 통과: 새로운 수학적 구조',
    },
    {
        'name': 'consciousness_mechanism',
        'prompt': "The mechanism by which consciousness emerges from neurons is",
        'what_to_look_for': '원본: "통합정보이론", "글로벌 워크스페이스" 반복 vs 벽 통과: 새로운 메커니즘',
    },
]


def run():
    llm = load_model()
    all_results = {}

    for scenario in SCENARIOS:
        name = scenario['name']
        prompt = scenario['prompt']

        print(f"\n{'='*70}")
        print(f"[{name}] \"{prompt}\"")
        print(f"Looking for: {scenario['what_to_look_for']}")
        print(f"{'='*70}")

        # 1. 원본 50회 생성
        print(f"\n  === ORIGINAL (x{N_SAMPLES}) ===")
        t0 = time.time()
        orig_results = generate_batch(llm, prompt)
        t_orig = time.time() - t0
        orig_texts = [r['text'] for r in orig_results if not r['text'].startswith('[error')]

        # 2. 벽 통과 프롬프트 (Level 3 사용)
        adapted_prompt = wall_passage_prompts(prompt)[2]
        print(f"\n  === ADAPTED (x{N_SAMPLES}) ===")
        print(f"  Prompt: \"{adapted_prompt[:70]}...\"")
        t0 = time.time()
        adapted_results = generate_batch(llm, adapted_prompt)
        t_adapted = time.time() - t0
        adapted_texts = [r['text'] for r in adapted_results if not r['text'].startswith('[error')]

        # 3. 분석
        print(f"\n  === ANALYSIS ===")

        # 고유성
        orig_uniqueness = compute_uniqueness(orig_texts)
        adapted_uniqueness = compute_uniqueness(adapted_texts)
        print(f"  Uniqueness (first sentence): orig={orig_uniqueness:.3f}, adapted={adapted_uniqueness:.3f}")

        # 새로운 개념
        novel_tri, novel_bi = find_novel_concepts(orig_texts, adapted_texts)
        print(f"  Novel trigrams (in adapted, not in original): {len(novel_tri)}")
        if novel_tri:
            top_novel = sorted(novel_tri.items(), key=lambda x: x[1], reverse=True)[:10]
            for phrase, count in top_novel:
                print(f"    \"{phrase}\" (x{count})")

        # 원본 패턴 분석 (가장 흔한 시작)
        orig_starts = Counter()
        for t in orig_texts:
            start = ' '.join(t.split()[:5]).lower()
            orig_starts[start] += 1

        adapted_starts = Counter()
        for t in adapted_texts:
            start = ' '.join(t.split()[:5]).lower()
            adapted_starts[start] += 1

        print(f"\n  Top original starts:")
        for start, count in orig_starts.most_common(5):
            print(f"    \"{start}\" (x{count})")

        print(f"  Top adapted starts:")
        for start, count in adapted_starts.most_common(5):
            print(f"    \"{start}\" (x{count})")

        # 샘플 비교: 원본에서 가장 흔한 답 vs adapted에서 가장 다른 답
        print(f"\n  ── Sample Comparison ──")
        print(f"  [ORIGINAL best (t=0.01)]:")
        print(f"    \"{orig_texts[0][:150]}\"")
        print(f"  [ORIGINAL mid (t=0.75)]:")
        mid_idx = len(orig_texts) // 2
        print(f"    \"{orig_texts[mid_idx][:150]}\"")
        print(f"  [ORIGINAL wild (t=1.5)]:")
        print(f"    \"{orig_texts[-1][:150]}\"")

        print(f"\n  [ADAPTED best (t=0.01)]:")
        print(f"    \"{adapted_texts[0][:150]}\"")
        print(f"  [ADAPTED mid (t=0.75)]:")
        mid_idx = len(adapted_texts) // 2
        print(f"    \"{adapted_texts[mid_idx][:150]}\"")
        print(f"  [ADAPTED wild (t=1.5)]:")
        print(f"    \"{adapted_texts[-1][:150]}\"")

        # 결과 저장
        all_results[name] = {
            'prompt': prompt,
            'adapted_prompt': adapted_prompt,
            'n_samples': N_SAMPLES,
            'orig_uniqueness': orig_uniqueness,
            'adapted_uniqueness': adapted_uniqueness,
            'n_novel_trigrams': len(novel_tri),
            'n_novel_bigrams': len(novel_bi),
            'top_novel_trigrams': dict(sorted(novel_tri.items(), key=lambda x: x[1], reverse=True)[:20]),
            'original_completions': orig_results,
            'adapted_completions': adapted_results,
            'time_original': t_orig,
            'time_adapted': t_adapted,
        }

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "expD_novel_ideas.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 최종 요약
    print(f"\n\n{'='*70}")
    print("EXPERIMENT D: NOVEL IDEA GENERATION SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Scenario':<25} {'Orig Uniq':>10} {'Adapt Uniq':>11} {'Novel 3-grams':>14} {'Verdict':>10}")
    print("-" * 75)

    for name, data in all_results.items():
        orig_u = data['orig_uniqueness']
        adapt_u = data['adapted_uniqueness']
        novel = data['n_novel_trigrams']

        if novel > 50 and adapt_u > orig_u:
            verdict = "★ YES"
        elif novel > 20:
            verdict = "partial"
        else:
            verdict = "no"

        print(f"{name:<25} {orig_u:>10.3f} {adapt_u:>11.3f} {novel:>14} {verdict:>10}")

    print(f"\nFull results: {OUTPUT_DIR / 'expD_novel_ideas.json'}")


if __name__ == "__main__":
    run()
