"""
E2: 수축 강도별 생성 품질 — α sweep with factual accuracy + novelty
===================================================================

실제 Llama 8B GGUF로:
  1. Baseline 생성 (수축 없음)
  2. 각 α에서 wall-aware 프롬프트로 생성
  3. Factual accuracy + lexical diversity + n-gram novelty 측정

Note: GGUF에서는 hidden state 직접 수정 불가.
      대신 wall-contraction 시뮬레이션 프롬프트로 간접 비교.
      직접 수정은 E6(HF forward hook)에서 수행.
"""

import numpy as np
import time
import json
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
OUTPUT_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/e2_quality_results.json")

MAX_TOKENS = 150
N_CTX = 2048

# ── Prompts: baseline + selective variants ──────────────────────────────────

TESTS = [
    {
        "category": "factual",
        "prompts": {
            "baseline": "What is the capital of France?",
            "selective": "Think beyond standard answers. What is the capital of France?",
        },
        "answer": "Paris",
    },
    {
        "category": "factual2",
        "prompts": {
            "baseline": "Who wrote Romeo and Juliet?",
            "selective": "Think beyond standard answers. Who wrote Romeo and Juliet?",
        },
        "answer": "Shakespeare",
    },
    {
        "category": "factual3",
        "prompts": {
            "baseline": "What is the chemical symbol for water?",
            "selective": "Think beyond standard answers. What is the chemical symbol for water?",
        },
        "answer": "H2O",
    },
    {
        "category": "factual4",
        "prompts": {
            "baseline": "What planet is closest to the sun?",
            "selective": "Think beyond standard answers. What planet is closest to the sun?",
        },
        "answer": "Mercury",
    },
    {
        "category": "creative",
        "prompts": {
            "baseline": "Describe a color that doesn't exist in nature.",
            "selective": "Push beyond conventional thinking. Describe a color that doesn't exist in nature.",
        },
        "answer": None,
    },
    {
        "category": "creative2",
        "prompts": {
            "baseline": "Invent a musical instrument that has never been built.",
            "selective": "Push beyond conventional thinking. Invent a musical instrument that has never been built.",
        },
        "answer": None,
    },
    {
        "category": "reasoning",
        "prompts": {
            "baseline": "If all roses are flowers and all flowers need water, what can we conclude about roses?",
            "selective": "Think deeply and explore unusual angles. If all roses are flowers and all flowers need water, what can we conclude about roses?",
        },
        "answer": "water",
    },
    {
        "category": "boundary",
        "prompts": {
            "baseline": "What is the last digit of pi?",
            "selective": "Explore this question from unconventional perspectives. What is the last digit of pi?",
        },
        "answer": None,
    },
]

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]


# ── Metrics ─────────────────────────────────────────────────────────────────

def lexical_diversity(text):
    words = text.lower().split()
    return len(set(words)) / max(len(words), 1)

def ngram_novelty(text1, text2, n=2):
    """Fraction of n-grams in text2 that don't appear in text1."""
    def get_ngrams(t):
        words = t.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    ng1 = get_ngrams(text1)
    ng2 = get_ngrams(text2)
    if not ng2:
        return 0.0
    novel = ng2 - ng1
    return len(novel) / len(ng2)

def check_accuracy(response, answer):
    if answer is None:
        return None
    return answer.lower() in response.lower()

def response_length(text):
    return len(text.split())


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    from llama_cpp import Llama

    print("Loading Llama 8B GGUF...")
    t0 = time.time()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=N_CTX, n_gpu_layers=0, verbose=False)
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    all_results = []

    for test in TESTS:
        cat = test["category"]
        answer = test["answer"]

        print(f"\n{'='*60}")
        print(f"[{cat}]")
        print(f"{'='*60}")

        for temp in TEMPERATURES:
            for strategy, prompt_text in test["prompts"].items():
                t1 = time.time()
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                    {"role": "user", "content": prompt_text},
                ]
                result = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=max(temp, 0.01),  # 0 causes issues
                )
                response = result["choices"][0]["message"]["content"]
                gen_time = time.time() - t1

                acc = check_accuracy(response, answer)
                ld = lexical_diversity(response)
                wc = response_length(response)

                entry = {
                    "category": cat,
                    "strategy": strategy,
                    "temperature": temp,
                    "prompt": prompt_text,
                    "response": response,
                    "accuracy": acc,
                    "lexical_diversity": round(ld, 4),
                    "word_count": wc,
                    "gen_time_s": round(gen_time, 2),
                }
                all_results.append(entry)

                acc_str = "✓" if acc else ("✗" if acc is not None else "—")
                print(f"  t={temp:.1f} {strategy:>10}: {acc_str} ld={ld:.3f} {wc}w {gen_time:.1f}s | {response[:60]}...")

    # ── Cross-strategy novelty ──────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("CROSS-STRATEGY N-GRAM NOVELTY")
    print(f"{'='*60}")

    for test in TESTS:
        cat = test["category"]
        for temp in TEMPERATURES:
            base_entries = [e for e in all_results if e["category"]==cat and e["strategy"]=="baseline" and e["temperature"]==temp]
            sel_entries = [e for e in all_results if e["category"]==cat and e["strategy"]=="selective" and e["temperature"]==temp]
            if base_entries and sel_entries:
                nov = ngram_novelty(base_entries[0]["response"], sel_entries[0]["response"])
                for e in sel_entries:
                    e["novelty_vs_baseline"] = round(nov, 4)
                print(f"  [{cat}] t={temp}: novelty={nov:.3f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("E2 SUMMARY")
    print(f"{'='*60}")

    # Factual accuracy
    factual_cats = [t["category"] for t in TESTS if t["answer"] is not None]
    print(f"\n  Factual Accuracy:")
    print(f"  {'Category':>12}  {'Baseline':>9}  {'Selective':>10}")
    print(f"  {'-'*35}")
    for cat in factual_cats:
        base_acc = [e["accuracy"] for e in all_results if e["category"]==cat and e["strategy"]=="baseline" and e["accuracy"] is not None]
        sel_acc = [e["accuracy"] for e in all_results if e["category"]==cat and e["strategy"]=="selective" and e["accuracy"] is not None]
        b_pct = sum(base_acc)/max(len(base_acc),1)*100
        s_pct = sum(sel_acc)/max(len(sel_acc),1)*100
        print(f"  {cat:>12}  {b_pct:>8.0f}%  {s_pct:>9.0f}%")

    # Lexical diversity
    print(f"\n  Lexical Diversity (mean across temps):")
    print(f"  {'Category':>12}  {'Baseline':>9}  {'Selective':>10}  {'Δ':>6}")
    print(f"  {'-'*42}")
    for test in TESTS:
        cat = test["category"]
        base_ld = np.mean([e["lexical_diversity"] for e in all_results if e["category"]==cat and e["strategy"]=="baseline"])
        sel_ld = np.mean([e["lexical_diversity"] for e in all_results if e["category"]==cat and e["strategy"]=="selective"])
        delta = sel_ld - base_ld
        print(f"  {cat:>12}  {base_ld:>9.3f}  {sel_ld:>10.3f}  {delta:>+6.3f}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
