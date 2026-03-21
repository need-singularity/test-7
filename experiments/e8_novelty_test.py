"""
E8: Novelty Verification — Is it new knowledge or just noise?
==============================================================
Same prompt, baseline 50x vs wall-removed 50x.
Measures: trigram novelty, semantic validity, repeatability.
"""

import numpy as np
import time
import json
import torch
from pathlib import Path
from collections import Counter

MODEL_ID = "C:/Users/aiden/models/llama-3.1-8b-instruct-hf"
OUTPUT_PATH = Path("e8_novelty_results.json")

CORE_WALL_DIMS = [782, 977, 1917, 1971, 2720, 2943, 3139, 4080]
WALL_RATE = 0.5  # contraction rate for wall dims

TEST_PROMPTS = [
    {"text": "Describe a color that doesn't exist in nature.", "category": "creative"},
    {"text": "What would a fifth fundamental force of nature do?", "category": "boundary"},
    {"text": "The solution to the Riemann hypothesis involves", "category": "knowledge"},
]

N_SAMPLES = 50
TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]


def get_trigrams(text):
    words = text.lower().split()
    return set(zip(words, words[1:], words[2:]))


def wall_contraction_hook(wall_dims, rate):
    """Forward hook that contracts wall dimensions toward their mean."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        center = hidden[:, :, wall_dims].mean(dim=1, keepdim=True).detach()
        direction = hidden[:, :, wall_dims] - center
        hidden[:, :, wall_dims] = hidden[:, :, wall_dims] - rate * direction
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def generate_samples(model, tokenizer, prompt_text, n, temperatures, device):
    """Generate n samples across different temperatures."""
    samples = []
    messages = [
        {"role": "system", "content": "You are a creative and knowledgeable assistant."},
        {"role": "user", "content": prompt_text},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for temp in temperatures:
        for _ in range(n // len(temperatures)):
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                )
            response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            samples.append({"text": response, "temperature": temp})
    return samples


def analyze_novelty(baseline_samples, adapted_samples):
    """Compare baseline vs adapted samples."""
    # Collect all baseline trigrams
    baseline_trigrams = set()
    for s in baseline_samples:
        baseline_trigrams.update(get_trigrams(s["text"]))

    # Find novel trigrams in adapted
    novel_trigrams = Counter()
    adapted_all_trigrams = set()
    for s in adapted_samples:
        tri = get_trigrams(s["text"])
        adapted_all_trigrams.update(tri)
        for t in tri:
            if t not in baseline_trigrams:
                novel_trigrams[t] += 1

    # Trigrams that appear 2+ times in adapted but never in baseline = stable novelty
    stable_novel = {t: c for t, c in novel_trigrams.items() if c >= 2}

    # Semantic validity: avg response length (short/broken = noise)
    baseline_lengths = [len(s["text"].split()) for s in baseline_samples]
    adapted_lengths = [len(s["text"].split()) for s in adapted_samples]

    # Unique responses (exact match dedup)
    baseline_unique = len(set(s["text"][:100] for s in baseline_samples))
    adapted_unique = len(set(s["text"][:100] for s in adapted_samples))

    return {
        "baseline_trigram_count": len(baseline_trigrams),
        "adapted_trigram_count": len(adapted_all_trigrams),
        "novel_trigram_count": len(novel_trigrams),
        "stable_novel_count": len(stable_novel),
        "top_stable_novel": [(" ".join(t), c) for t, c in
                             sorted(stable_novel.items(), key=lambda x: -x[1])[:20]],
        "baseline_avg_length": float(np.mean(baseline_lengths)),
        "adapted_avg_length": float(np.mean(adapted_lengths)),
        "baseline_unique_responses": baseline_unique,
        "adapted_unique_responses": adapted_unique,
        "overlap_ratio": len(baseline_trigrams & adapted_all_trigrams) / max(len(baseline_trigrams), 1),
    }


def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (no GPU)")
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        print(f"Device: {device} (manual)")
        print("\nLoading model (fp16 on CPU)...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    else:
        device = "cuda"
        print(f"Device: {device}")
        print("\nLoading model (4-bit on GPU)...")
        t0 = time.time()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = {}

    for prompt_info in TEST_PROMPTS:
        prompt_text = prompt_info["text"]
        category = prompt_info["category"]
        print(f"\n== [{category}] {prompt_text}")

        # Phase 1: Baseline samples (no hook)
        print(f"  Generating {N_SAMPLES} baseline samples...")
        model.eval()
        t1 = time.time()
        baseline_samples = generate_samples(model, tokenizer, prompt_text, N_SAMPLES, TEMPERATURES, device)
        print(f"    Done in {time.time()-t1:.1f}s")

        # Phase 2: Wall-removed samples (with hook on last layer)
        print(f"  Generating {N_SAMPLES} wall-removed samples...")
        last_layer = model.model.layers[-1]
        hook_handle = last_layer.register_forward_hook(
            wall_contraction_hook(CORE_WALL_DIMS, WALL_RATE)
        )
        t2 = time.time()
        adapted_samples = generate_samples(model, tokenizer, prompt_text, N_SAMPLES, TEMPERATURES, device)
        hook_handle.remove()
        print(f"    Done in {time.time()-t2:.1f}s")

        # Phase 3: Analyze
        analysis = analyze_novelty(baseline_samples, adapted_samples)

        # Verdict
        stable = analysis["stable_novel_count"]
        avg_len = analysis["adapted_avg_length"]
        if stable > 50 and avg_len > 20:
            verdict = "NEW_KNOWLEDGE"
        elif stable > 20 and avg_len > 20:
            verdict = "PARTIAL_NOVELTY"
        elif avg_len < 10:
            verdict = "NOISE_OOD_COLLAPSE"
        elif stable < 5:
            verdict = "NO_EFFECT"
        else:
            verdict = "INCONCLUSIVE"

        print(f"\n  Results:")
        print(f"    Baseline trigrams: {analysis['baseline_trigram_count']}")
        print(f"    Adapted trigrams:  {analysis['adapted_trigram_count']}")
        print(f"    Novel trigrams:    {analysis['novel_trigram_count']}")
        print(f"    Stable novel (2+): {analysis['stable_novel_count']}")
        print(f"    Overlap ratio:     {analysis['overlap_ratio']:.3f}")
        print(f"    Baseline avg len:  {analysis['baseline_avg_length']:.1f} words")
        print(f"    Adapted avg len:   {analysis['adapted_avg_length']:.1f} words")
        print(f"    Baseline unique:   {analysis['baseline_unique_responses']}/{N_SAMPLES}")
        print(f"    Adapted unique:    {analysis['adapted_unique_responses']}/{N_SAMPLES}")
        print(f"    VERDICT: {verdict}")

        if analysis["top_stable_novel"]:
            print(f"    Top novel trigrams:")
            for tri, count in analysis["top_stable_novel"][:5]:
                print(f"      [{count}x] {tri}")

        results[category] = {
            "prompt": prompt_text,
            "analysis": analysis,
            "verdict": verdict,
            "baseline_samples": [s["text"][:200] for s in baseline_samples[:5]],
            "adapted_samples": [s["text"][:200] for s in adapted_samples[:5]],
        }

    # Summary
    print("\n\n" + "=" * 60)
    print("E8 SUMMARY - Novelty Verification")
    print("=" * 60)
    for cat, r in results.items():
        a = r["analysis"]
        print(f"  [{cat}] {r['verdict']}")
        print(f"    stable novel: {a['stable_novel_count']}, overlap: {a['overlap_ratio']:.3f}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
