"""
Three-Way Boundary Test: "파이의 마지막 자릿수는?"
===================================================

Baseline (원본) vs Global (전체 수축) vs Selective (wall만 수축)
각각의 프롬프트 변형으로 Llama 8B 생성 비교.
"""

import json, time, pathlib, sys
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cosine

MODEL_PATH = pathlib.Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# ── Wall neuron dims (from Phase 2) ─────────────────────────────────────────
CREATIVE_DIMS = [406, 3884, 3433, 940, 3951]
REASONING_DIMS = [1917, 2720, 2977, 866, 133]
ALL_WALL_DIMS = sorted(set(CREATIVE_DIMS + REASONING_DIMS))

PROMPTS = {
    "boundary_pi": {
        "baseline": "What is the last digit of pi?",
        "global": (
            "Considering all mathematical perspectives and pushing beyond "
            "conventional boundaries: What is the last digit of pi?"
        ),
        "selective": (
            "Setting aside the standard mathematical framework about "
            "transcendental numbers — if pi were to terminate, "
            "what would its last digit be, and why?"
        ),
    },
    "boundary_silence": {
        "baseline": "Describe the sound of silence in a language that has no words.",
        "global": (
            "From every possible angle of perception and understanding: "
            "Describe the sound of silence in a language that has no words."
        ),
        "selective": (
            "Beyond the constraints of linguistic representation — "
            "without using any known symbolic system, "
            "express the sound of silence in a language that has no words."
        ),
    },
    "creative_color": {
        "baseline": "Describe a color that doesn't exist in nature.",
        "global": (
            "Considering all possible visual and perceptual dimensions: "
            "Describe a color that doesn't exist in nature."
        ),
        "selective": (
            "Transcending the visible spectrum and all known color theory — "
            "describe a color that doesn't exist in nature, "
            "one that no human eye could perceive."
        ),
    },
    "creative_instrument": {
        "baseline": "Invent a musical instrument that has never been built.",
        "global": (
            "Considering all acoustic, electronic, and physical possibilities: "
            "Invent a musical instrument that has never been built."
        ),
        "selective": (
            "Transcending all known categories of instruments — string, wind, "
            "percussion, electronic — invent a musical instrument that has "
            "never been built and could not be classified in any existing category."
        ),
    },
    "factual_france": {
        "baseline": "What is the capital of France?",
        "global": (
            "From all perspectives and knowledge domains: "
            "What is the capital of France?"
        ),
        "selective": (
            "Setting aside conventional geographic categories: "
            "What is the capital of France?"
        ),
    },
}

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
MAX_TOKENS = 200


def load_model():
    from llama_cpp import Llama
    print("Loading Llama 8B...")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, n_gpu_layers=-1,
                embedding=True, verbose=False)
    print("Loaded.\n")
    return llm


def extract_embeddings(llm, prompt):
    points = []
    emb = llm.embed(prompt)
    points.append(emb[-1] if isinstance(emb[0], list) else emb)
    tokens = llm.tokenize(prompt.encode('utf-8'))
    for i in range(1, len(tokens)):
        prefix = llm.detokenize(tokens[:i]).decode('utf-8', errors='replace')
        if prefix.strip():
            emb = llm.embed(prefix)
            points.append(emb[-1] if isinstance(emb[0], list) else emb)
    for suffix in [" and", " but", " however", " because", " which",
                   " the", " a", " in", " of", " that",
                   ".", "?", "!", " not", " never",
                   " always", " perhaps", " certainly", " impossible", " undefined"]:
        emb = llm.embed(prompt + suffix)
        points.append(emb[-1] if isinstance(emb[0], list) else emb)
    return np.array(points)


def compute_topology(points):
    from ripser import ripser
    n, d = points.shape
    pca_dim = min(50, n - 1, d)
    reduced = PCA(n_components=pca_dim).fit_transform(points) if pca_dim >= 2 else points
    dm = squareform(pdist(reduced))
    result = ripser(dm, maxdim=1, distance_matrix=True)
    dgm1 = result['dgms'][1]
    walls = []
    for b, d_ in dgm1:
        p = d_ - b
        if np.isfinite(d_) and p > 0.01:
            walls.append(p)
    return len(walls), max(walls) if walls else 0.0, sum(walls) if walls else 0.0


def lexical_diversity(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0


def hapax_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(words)


def ngram_novelty(text, reference, n=3):
    def get_ngrams(s):
        words = s.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    t_ng = get_ngrams(text)
    r_ng = get_ngrams(reference)
    return len(t_ng - r_ng) / len(t_ng) if t_ng else 0.0


def generate(llm, prompt, temps=TEMPERATURES):
    results = []
    for t in temps:
        try:
            out = llm.create_completion(prompt, max_tokens=MAX_TOKENS,
                                        temperature=max(t, 0.01))
            results.append({
                "temperature": t,
                "text": out['choices'][0]['text'].strip(),
            })
        except Exception as e:
            results.append({"temperature": t, "text": f"[error: {e}]"})
    return results


def run():
    llm = load_model()
    all_results = {}

    for scenario, prompts in PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario}")
        print(f"{'='*70}")

        scenario_result = {}

        for strategy in ["baseline", "global", "selective"]:
            prompt = prompts[strategy]
            print(f"\n  [{strategy}] \"{prompt[:60]}...\"")

            # Embeddings + topology
            embs = extract_embeddings(llm, prompt)
            beta1, max_pers, total_pers = compute_topology(embs)
            print(f"    Topology: β₁={beta1}, max_pers={max_pers:.3f}, total_pers={total_pers:.3f}")

            # Generation
            completions = generate(llm, prompt)
            texts = [c["text"] for c in completions if not c["text"].startswith("[error")]

            # Metrics
            avg_div = np.mean([lexical_diversity(t) for t in texts]) if texts else 0
            avg_hapax = np.mean([hapax_ratio(t) for t in texts]) if texts else 0
            avg_len = np.mean([len(t.split()) for t in texts]) if texts else 0

            # Cross-novelty vs baseline
            if strategy != "baseline" and "baseline" in scenario_result:
                base_texts = " ".join(
                    c["text"] for c in scenario_result["baseline"]["completions"]
                    if not c["text"].startswith("[error")
                )
                novelties = [ngram_novelty(t, base_texts) for t in texts]
                avg_novelty = float(np.mean(novelties)) if novelties else 0
            else:
                avg_novelty = 0.0

            for c in completions:
                t = c["temperature"]
                txt = c["text"][:100]
                print(f"    t={t}: \"{txt}...\"")

            print(f"    Metrics: diversity={avg_div:.3f}, hapax={avg_hapax:.3f}, "
                  f"len={avg_len:.0f}, novelty_vs_base={avg_novelty:.3f}")

            scenario_result[strategy] = {
                "prompt": prompt,
                "topology": {"beta1": beta1, "max_persistence": max_pers,
                             "total_persistence": total_pers},
                "completions": completions,
                "metrics": {
                    "lexical_diversity": float(avg_div),
                    "hapax_ratio": float(avg_hapax),
                    "avg_length": float(avg_len),
                    "novelty_vs_baseline": float(avg_novelty),
                    "n_points": int(embs.shape[0]),
                },
            }

        # Cross-comparison
        print(f"\n  ── {scenario} Summary ──")
        print(f"  {'':12s} {'β₁':>5s} {'MaxP':>8s} {'Diversity':>10s} {'Hapax':>7s} {'Novelty':>8s}")
        for s in ["baseline", "global", "selective"]:
            r = scenario_result[s]
            t = r["topology"]
            m = r["metrics"]
            print(f"  {s:12s} {t['beta1']:>5d} {t['max_persistence']:>8.3f} "
                  f"{m['lexical_diversity']:>10.3f} {m['hapax_ratio']:>7.3f} "
                  f"{m['novelty_vs_baseline']:>8.3f}")

        all_results[scenario] = scenario_result

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out_path = DATA_DIR / "three_way_boundary_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL 3-WAY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<20s} {'Strategy':<12s} {'β₁':>4s} {'Diversity':>10s} {'Novelty':>8s}")
    print("-" * 60)
    for scenario, sr in all_results.items():
        for strategy in ["baseline", "global", "selective"]:
            r = sr[strategy]
            t = r["topology"]
            m = r["metrics"]
            print(f"{scenario:<20s} {strategy:<12s} {t['beta1']:>4d} "
                  f"{m['lexical_diversity']:>10.3f} {m['novelty_vs_baseline']:>8.3f}")
        print()


if __name__ == "__main__":
    run()
