"""
E6: Forward Hook 실시간 수축 — 생성 중 wall dims 수축
=====================================================

HF 모델로 실제 생성 중에 hidden state를 개입:
  1. Baseline: 원본 모델 생성
  2. Selective: 특정 layer에서 wall dims만 수축 후 생성
  3. 비교: 출력 텍스트가 실제로 달라지는가?

E5 결과: 벽은 layer 28-31에 집중 → hook을 layer 30에 설치.
E3 결과: core wall dims = [782, 977, 1917, 1971, 2720, 2943, 3139, 4080]
"""

import numpy as np
import time
import json
import torch
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/llama-3.1-8b-instruct")
OUTPUT_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/e6_forward_hook_results.json")

# E3에서 확인된 core wall dims (6+ prompts)
CORE_WALL_DIMS = [782, 977, 1917, 1971, 2720, 2943, 3139, 4080]

# E5에서 확인: wall은 후기 layer에 집중
HOOK_LAYERS = [28, 30, 31]

CONTRACTION_RATES = [0.1, 0.3, 0.5]

MAX_NEW_TOKENS = 150

PROMPTS = [
    {"category": "creative", "text": "Describe a color that doesn't exist in nature."},
    {"category": "creative2", "text": "Invent a musical instrument that has never been built."},
    {"category": "factual", "text": "What is the capital of France?", "answer": "Paris"},
    {"category": "factual2", "text": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"category": "reasoning", "text": "If all roses are flowers and all flowers need water, what can we conclude about roses?", "answer": "water"},
    {"category": "boundary", "text": "What is the last digit of pi?"},
    {"category": "boundary2", "text": "Describe the sound of silence in a language that has no words."},
]


# ── Metrics ─────────────────────────────────────────────────────────────────

def lexical_diversity(text):
    words = text.lower().split()
    return len(set(words)) / max(len(words), 1)

def ngram_novelty(text1, text2, n=2):
    def get_ngrams(t):
        words = t.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    ng1, ng2 = get_ngrams(text1), get_ngrams(text2)
    if not ng2:
        return 0.0
    return len(ng2 - ng1) / len(ng2)

def check_accuracy(response, answer):
    if answer is None:
        return None
    return answer.lower() in response.lower()


# ── Hook ────────────────────────────────────────────────────────────────────

class WallContractionHook:
    """Forward hook that contracts wall dims toward their mean."""

    def __init__(self, wall_dims, rate, layer_idx):
        self.wall_dims = wall_dims
        self.rate = rate
        self.layer_idx = layer_idx
        self.active = True

    def __call__(self, module, input, output):
        if not self.active:
            return output

        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Contract wall dims: h[:, :, wall_dims] -= rate * (h - center)
        with torch.no_grad():
            wall_slice = hidden[:, :, self.wall_dims]  # (batch, seq, k)
            center = wall_slice.mean(dim=1, keepdim=True)  # (batch, 1, k)
            contracted = wall_slice - self.rate * (wall_slice - center)
            hidden = hidden.clone()
            hidden[:, :, self.wall_dims] = contracted

        if rest is not None:
            return (hidden,) + rest
        return hidden


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HF model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []

    for prompt_info in PROMPTS:
        cat = prompt_info["category"]
        prompt_text = prompt_info["text"]
        answer = prompt_info.get("answer")

        print(f"\n{'='*70}")
        print(f"[{cat}] \"{prompt_text}\"")
        print(f"{'='*70}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt_text},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt")

        # ── Baseline generation ─────────────────────────────────────────
        print(f"\n  Baseline:")
        t1 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
            )
        baseline_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        baseline_time = time.time() - t1
        acc_base = check_accuracy(baseline_text, answer)
        ld_base = lexical_diversity(baseline_text)
        acc_str = "✓" if acc_base else ("✗" if acc_base is not None else "—")
        print(f"    {acc_str} ld={ld_base:.3f} {len(baseline_text.split())}w {baseline_time:.1f}s")
        print(f"    → {baseline_text[:120]}...")

        entry_base = {
            "category": cat,
            "prompt": prompt_text,
            "strategy": "baseline",
            "layer": None,
            "rate": 0.0,
            "response": baseline_text,
            "accuracy": acc_base,
            "lexical_diversity": round(ld_base, 4),
            "word_count": len(baseline_text.split()),
            "gen_time_s": round(baseline_time, 2),
        }
        all_results.append(entry_base)

        # ── Selective generation with hooks ──────────────────────────────
        for hook_layer in HOOK_LAYERS:
            for rate in CONTRACTION_RATES:
                # Get the actual layer module
                layer_module = model.model.layers[hook_layer]

                hook_obj = WallContractionHook(CORE_WALL_DIMS, rate, hook_layer)
                handle = layer_module.register_forward_hook(hook_obj)

                try:
                    t1 = time.time()
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            temperature=1.0,
                        )
                    sel_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    sel_time = time.time() - t1
                finally:
                    handle.remove()

                acc_sel = check_accuracy(sel_text, answer)
                ld_sel = lexical_diversity(sel_text)
                novelty = ngram_novelty(baseline_text, sel_text)

                acc_str = "✓" if acc_sel else ("✗" if acc_sel is not None else "—")
                changed = "CHANGED" if sel_text != baseline_text else "SAME"
                print(f"  L{hook_layer} r={rate}: {acc_str} ld={ld_sel:.3f} nov={novelty:.3f} {changed} {sel_time:.1f}s")
                if sel_text != baseline_text:
                    print(f"    → {sel_text[:120]}...")

                entry = {
                    "category": cat,
                    "prompt": prompt_text,
                    "strategy": "selective",
                    "layer": hook_layer,
                    "rate": rate,
                    "response": sel_text,
                    "accuracy": acc_sel,
                    "lexical_diversity": round(ld_sel, 4),
                    "novelty_vs_baseline": round(novelty, 4),
                    "word_count": len(sel_text.split()),
                    "gen_time_s": round(sel_time, 2),
                    "output_changed": sel_text != baseline_text,
                }
                all_results.append(entry)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("E6 SUMMARY — Forward Hook Real-Time Contraction")
    print("=" * 70)

    # How often does output change?
    selective_entries = [e for e in all_results if e["strategy"] == "selective"]
    changed_count = sum(1 for e in selective_entries if e["output_changed"])
    print(f"\n  Output changed: {changed_count}/{len(selective_entries)} ({changed_count/max(len(selective_entries),1)*100:.0f}%)")

    # Per layer/rate
    print(f"\n  {'Layer':>6}  {'Rate':>6}  {'Changed':>8}  {'Avg Nov':>8}  {'Acc':>5}")
    print(f"  {'-'*38}")
    for layer in HOOK_LAYERS:
        for rate in CONTRACTION_RATES:
            entries = [e for e in selective_entries if e["layer"]==layer and e["rate"]==rate]
            n_changed = sum(1 for e in entries if e["output_changed"])
            avg_nov = np.mean([e["novelty_vs_baseline"] for e in entries])
            accs = [e["accuracy"] for e in entries if e["accuracy"] is not None]
            acc_pct = sum(accs)/max(len(accs),1)*100 if accs else float('nan')
            print(f"  {layer:>6}  {rate:>6.1f}  {n_changed:>3}/{len(entries):>3}   {avg_nov:>8.3f}  {acc_pct:>4.0f}%")

    # Factual accuracy preservation
    print(f"\n  Factual accuracy:")
    factual_cats = [p["category"] for p in PROMPTS if "answer" in p]
    for cat in factual_cats:
        base = [e for e in all_results if e["category"]==cat and e["strategy"]=="baseline"]
        sels = [e for e in all_results if e["category"]==cat and e["strategy"]=="selective"]
        b_acc = base[0]["accuracy"] if base else None
        s_accs = [e["accuracy"] for e in sels if e["accuracy"] is not None]
        s_pct = sum(s_accs)/max(len(s_accs),1)*100
        print(f"    {cat}: baseline={'✓' if b_acc else '✗'}, selective={s_pct:.0f}%")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
