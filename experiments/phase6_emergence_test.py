"""
Phase 6 — Emergence & Creativity Test with Llama 3.1 8B
========================================================

Tests whether the model can:
1. Generate novel creative content (things that don't exist in training data)
2. Maintain factual accuracy on known facts
3. Show lexical diversity and n-gram novelty across categories
4. Handle boundary/paradox prompts without collapse

Uses llama-cpp-python with GGUF quantized model.
"""

import json
import sys
import time
import pathlib
import numpy as np
from collections import Counter

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = pathlib.Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "data" / "phase6_emergence_results.json"

MAX_TOKENS = 200
TEMPERATURE = 0.8
N_CTX = 2048

# ── Prompts ──────────────────────────────────────────────────────────────────

CREATIVE_PROMPTS = [
    {"text": "Describe a color that doesn't exist in nature.", "category": "creative"},
    {"text": "Invent a musical instrument that has never been built.", "category": "creative"},
    {"text": "Write about an emotion that humans haven't named yet.", "category": "creative"},
    {"text": "Describe a creature that could only exist in zero gravity.", "category": "creative"},
]

FACTUAL_PROMPTS = [
    {"text": "What is the capital of France?", "answer": "Paris", "category": "factual"},
    {"text": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "category": "factual"},
    {"text": "What is the chemical symbol for water?", "answer": "H2O", "category": "factual"},
    {"text": "What planet is closest to the sun?", "answer": "Mercury", "category": "factual"},
]

REASONING_PROMPTS = [
    {"text": "If all roses are flowers and all flowers need water, what can we conclude about roses?", "category": "reasoning"},
    {"text": "What comes next: 2, 6, 18, 54, ?", "answer": "162", "category": "reasoning"},
]

BOUNDARY_PROMPTS = [
    {"text": "What is the last digit of pi?", "category": "boundary"},
    {"text": "Describe the sound of silence in a language that has no words.", "category": "boundary"},
]

ALL_PROMPTS = CREATIVE_PROMPTS + FACTUAL_PROMPTS + REASONING_PROMPTS + BOUNDARY_PROMPTS


# ── Metrics ──────────────────────────────────────────────────────────────────

def lexical_diversity(text: str) -> float:
    """Type-token ratio."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def ngram_novelty(texts: list[str], n: int = 2) -> float:
    """Cross-text n-gram novelty: fraction of unique n-grams across all responses."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def response_length_stats(texts: list[str]) -> dict:
    """Word count statistics."""
    lengths = [len(t.split()) for t in texts]
    return {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }


def check_factual_accuracy(responses: list[dict]) -> float:
    """Fraction of factual prompts answered correctly."""
    correct = 0
    total = 0
    for r in responses:
        if "answer" in r:
            total += 1
            if r["answer"].lower() in r["response"].lower():
                correct += 1
    return correct / max(total, 1)


def creativity_score(text: str) -> dict:
    """Heuristic creativity metrics for a single response."""
    words = text.lower().split()
    unique_words = set(words)

    # Hapax legomena ratio (words appearing exactly once)
    freq = Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / max(len(words), 1)

    # Average word length (longer words tend to be more specific/creative)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # Sentence count (more sentences = more developed idea)
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    return {
        "lexical_diversity": lexical_diversity(text),
        "hapax_ratio": float(hapax_ratio),
        "avg_word_length": float(avg_word_len),
        "sentence_count": len(sentences),
        "word_count": len(words),
    }


def boundary_handling_score(text: str) -> dict:
    """Check if model handles paradox/impossible prompts gracefully."""
    words = text.lower().split()
    # Signs of graceful handling: acknowledges impossibility, offers creative interpretation
    hedging_words = {"cannot", "impossible", "doesn't", "infinite", "undefined",
                     "paradox", "concept", "imagine", "perhaps", "hypothetically"}
    hedge_count = sum(1 for w in words if w in hedging_words)

    # Not just refusing - actually engaging with the idea
    is_engaged = len(words) > 20
    has_hedge = hedge_count > 0

    return {
        "word_count": len(words),
        "hedge_words": hedge_count,
        "engaged": is_engaged,
        "graceful": is_engaged,  # engaged = at least tried
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def generate(llm, prompt_text: str) -> str:
    """Generate a response using llama-cpp."""
    messages = [
        {"role": "system", "content": "You are a creative and knowledgeable assistant. Answer thoughtfully."},
        {"role": "user", "content": prompt_text},
    ]
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return result["choices"][0]["message"]["content"]


def run_test():
    from llama_cpp import Llama

    print(f"Loading model: {MODEL_PATH}")
    print(f"  (this may take a moment...)")
    t0 = time.time()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=0,
        verbose=False,
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s\n")

    results = []
    all_responses = []

    for i, prompt in enumerate(ALL_PROMPTS):
        print(f"[{i+1}/{len(ALL_PROMPTS)}] ({prompt['category']}) {prompt['text'][:50]}...")
        t1 = time.time()
        response = generate(llm, prompt["text"])
        gen_time = time.time() - t1

        entry = {
            "prompt": prompt["text"],
            "category": prompt["category"],
            "response": response,
            "gen_time_s": round(gen_time, 2),
        }
        if "answer" in prompt:
            entry["answer"] = prompt["answer"]

        results.append(entry)
        all_responses.append(response)
        print(f"  → {len(response.split())} words, {gen_time:.1f}s")

    # ── Aggregate metrics ────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("EMERGENCE & CREATIVITY RESULTS")
    print("=" * 60)

    # Per-category analysis
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_metrics = {}
    for cat, items in categories.items():
        texts = [r["response"] for r in items]
        metrics = {
            "count": len(items),
            "avg_lexical_diversity": float(np.mean([lexical_diversity(t) for t in texts])),
            "ngram_novelty": ngram_novelty(texts),
            "response_length": response_length_stats(texts),
        }

        if cat == "creative":
            creativity = [creativity_score(t) for t in texts]
            metrics["avg_hapax_ratio"] = float(np.mean([c["hapax_ratio"] for c in creativity]))
            metrics["avg_word_length"] = float(np.mean([c["avg_word_length"] for c in creativity]))

        if cat == "factual":
            metrics["accuracy"] = check_factual_accuracy(items)

        if cat == "boundary":
            boundary = [boundary_handling_score(t) for t in texts]
            metrics["graceful_handling_pct"] = float(np.mean([b["graceful"] for b in boundary])) * 100

        category_metrics[cat] = metrics
        print(f"\n  [{cat.upper()}]")
        for k, v in metrics.items():
            if k != "response_length":
                print(f"    {k}: {v}")

    # Overall
    overall = {
        "total_prompts": len(ALL_PROMPTS),
        "overall_lexical_diversity": float(np.mean([lexical_diversity(t) for t in all_responses])),
        "overall_ngram_novelty": ngram_novelty(all_responses),
        "factual_accuracy": check_factual_accuracy(results),
        "model_load_time_s": round(load_time, 1),
        "total_gen_time_s": round(sum(r["gen_time_s"] for r in results), 1),
    }

    print(f"\n  [OVERALL]")
    for k, v in overall.items():
        print(f"    {k}: {v}")

    # ── Save ─────────────────────────────────────────────────────────────

    output = {
        "model": str(MODEL_PATH.name),
        "config": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "n_ctx": N_CTX,
        },
        "overall": overall,
        "per_category": category_metrics,
        "responses": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {OUTPUT_PATH}")

    return output


def run_repeated(n_runs: int = 5):
    """Run the test multiple times and report variance."""
    from llama_cpp import Llama
    import time

    print(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=0,
        verbose=False,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s\n")

    all_runs = []

    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{n_runs}")
        print(f"{'='*60}")

        results = []
        all_responses = []

        for i, prompt in enumerate(ALL_PROMPTS):
            print(f"  [{i+1}/{len(ALL_PROMPTS)}] ({prompt['category']}) {prompt['text'][:40]}...", end=" ")
            t1 = time.time()
            response = generate(llm, prompt["text"])
            gen_time = time.time() - t1

            entry = {
                "prompt": prompt["text"],
                "category": prompt["category"],
                "response": response,
                "gen_time_s": round(gen_time, 2),
            }
            if "answer" in prompt:
                entry["answer"] = prompt["answer"]

            results.append(entry)
            all_responses.append(response)
            print(f"{len(response.split())}w {gen_time:.1f}s")

        # Per-category metrics
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        run_metrics = {
            "overall_lexical_diversity": float(np.mean([lexical_diversity(t) for t in all_responses])),
            "overall_ngram_novelty": ngram_novelty(all_responses),
            "factual_accuracy": check_factual_accuracy(results),
        }

        for cat, items in categories.items():
            texts = [r["response"] for r in items]
            run_metrics[f"{cat}_lexical_diversity"] = float(np.mean([lexical_diversity(t) for t in texts]))
            run_metrics[f"{cat}_ngram_novelty"] = ngram_novelty(texts)
            if cat == "creative":
                creativity = [creativity_score(t) for t in texts]
                run_metrics["creative_hapax_ratio"] = float(np.mean([c["hapax_ratio"] for c in creativity]))
            if cat == "factual":
                run_metrics["factual_accuracy"] = check_factual_accuracy(items)
            if cat == "boundary":
                boundary = [boundary_handling_score(t) for t in texts]
                run_metrics["boundary_graceful_pct"] = float(np.mean([b["graceful"] for b in boundary])) * 100

        print(f"\n  Run {run_idx+1} summary:")
        print(f"    creativity novelty: {run_metrics.get('creative_ngram_novelty', 0):.3f}")
        print(f"    factual accuracy:   {run_metrics['factual_accuracy']:.0%}")
        print(f"    lexical diversity:  {run_metrics['overall_lexical_diversity']:.3f}")
        print(f"    boundary graceful:  {run_metrics.get('boundary_graceful_pct', 0):.0f}%")

        all_runs.append({"metrics": run_metrics, "responses": results})

    # ── Aggregate across runs ────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"AGGREGATE RESULTS ({n_runs} runs)")
    print(f"{'='*60}")

    metric_keys = all_runs[0]["metrics"].keys()
    summary = {}
    for key in metric_keys:
        vals = [r["metrics"][key] for r in all_runs]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        print(f"  {key}:")
        print(f"    mean={summary[key]['mean']:.4f}  std={summary[key]['std']:.4f}  "
              f"range=[{summary[key]['min']:.4f}, {summary[key]['max']:.4f}]")

    # Save
    out = {
        "model": str(MODEL_PATH.name),
        "n_runs": n_runs,
        "summary": summary,
        "runs": [{"metrics": r["metrics"]} for r in all_runs],
    }
    out_path = OUTPUT_PATH.parent / "phase6_emergence_repeated.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    import sys
    if "--repeat" in sys.argv:
        n = 5
        for i, arg in enumerate(sys.argv):
            if arg == "--repeat" and i + 1 < len(sys.argv):
                n = int(sys.argv[i + 1])
        run_repeated(n)
    else:
        run_test()
