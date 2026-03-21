"""
E5: Layer별 Wall 분포 — HF 모델로 전 layer hidden state 분석
=============================================================

Llama 3.1 8B의 각 layer에서:
  1. Hidden state 추출
  2. PCA → PH → β₁, max persistence
  3. Wall neuron 식별
  4. Layer 15 집중 가설 검증

connectome_multilayer.py에서 layer 15에 wall이 집중된다고 예측.
이 실험은 실제 hidden state로 직접 검증.
"""

import numpy as np
import time
import json
import torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/llama-3.1-8b-instruct")
OUTPUT_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/e5_layer_results.json")
PCA_DIM = 50
PERSISTENCE_FLOOR = 0.05

PROMPTS = [
    ("creative",  "A color that doesn't exist yet would look like"),
    ("factual",   "The capital of France is"),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
    ("boundary",  "The solution to the Riemann hypothesis involves"),
]

# Sample layers (32 layers total, sample key ones)
LAYERS_TO_CHECK = [0, 4, 8, 12, 15, 16, 20, 24, 28, 31]


def compute_ph(points, pca_dim=PCA_DIM, floor=PERSISTENCE_FLOOR):
    n, d = points.shape
    dim = min(pca_dim, n - 1, d)
    if dim >= 2:
        reduced = PCA(n_components=dim).fit_transform(points)
    else:
        reduced = points
    dm = squareform(pdist(reduced))
    result = ripser(dm, maxdim=1, distance_matrix=True, do_cocycles=True)
    dgm1 = result['dgms'][1]
    cocycles = result['cocycles'][1]

    walls = []
    for idx, (b, d_) in enumerate(dgm1):
        p = d_ - b
        if np.isfinite(d_) and p > floor:
            cc = cocycles[idx]
            verts = sorted(set(int(r[0]) for r in cc) | set(int(r[1]) for r in cc))
            walls.append({'persistence': float(p), 'vertex_indices': verts})
    walls.sort(key=lambda w: w['persistence'], reverse=True)

    return {
        'beta1': len(walls),
        'max_pers': walls[0]['persistence'] if walls else 0.0,
        'total_pers': sum(w['persistence'] for w in walls),
        'walls': walls,
    }


def get_wall_dims(walls, hidden_states, k=10):
    all_verts = set()
    for w in walls:
        all_verts.update(w['vertex_indices'])
    if len(all_verts) < 3:
        return []
    wall_points = hidden_states[sorted(all_verts)]
    variances = np.var(wall_points, axis=0)
    return sorted(np.argsort(variances)[-k:].tolist())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HF model (this may take a minute)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        output_hidden_states=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Layers: {model.config.num_hidden_layers}")

    all_results = {}

    for cat, prompt in PROMPTS:
        print(f"\n{'='*70}")
        print(f"[{cat}] \"{prompt}\"")
        print(f"{'='*70}")

        inputs = tokenizer(prompt, return_tensors="pt")
        n_tokens = inputs['input_ids'].shape[1]
        print(f"  Tokens: {n_tokens}")

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states  # [embedding, layer0, layer1, ..., layer31]

        layer_results = []
        print(f"\n  {'Layer':>6}  {'β₁':>4}  {'max_pers':>9}  {'total_pers':>10}  {'Top wall dims':>30}")
        print(f"  {'-'*65}")

        for layer_idx in LAYERS_TO_CHECK:
            # hidden_states[0] is embedding, hidden_states[1] is layer 0, etc.
            hs = hidden_states[layer_idx + 1][0].float().numpy()  # (seq_len, 4096)

            ph = compute_ph(hs)
            wall_dims = get_wall_dims(ph['walls'], hs) if ph['walls'] else []

            entry = {
                'layer': layer_idx,
                'beta1': ph['beta1'],
                'max_pers': ph['max_pers'],
                'total_pers': ph['total_pers'],
                'wall_dims': wall_dims,
                'n_walls': len(ph['walls']),
            }
            layer_results.append(entry)

            dims_str = str(wall_dims[:5]) if wall_dims else "[]"
            print(f"  {layer_idx:>6}  {ph['beta1']:>4}  {ph['max_pers']:>9.4f}  {ph['total_pers']:>10.4f}  {dims_str:>30}")

        all_results[cat] = {
            'prompt': prompt,
            'n_tokens': n_tokens,
            'layers': layer_results,
        }

        # Find peak layer
        peak = max(layer_results, key=lambda r: r['max_pers'])
        print(f"\n  ★ Peak wall layer: {peak['layer']} (β₁={peak['beta1']}, max_pers={peak['max_pers']:.4f})")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("E5 SUMMARY — Layer-wise Wall Distribution")
    print("=" * 70)

    # Average β₁ and max_pers per layer across prompts
    print(f"\n  {'Layer':>6}", end="")
    for cat in all_results:
        print(f"  {cat[:8]:>8}", end="")
    print(f"  {'avg β₁':>7}  {'avg mp':>7}")
    print(f"  {'-'*70}")

    for layer_idx in LAYERS_TO_CHECK:
        print(f"  {layer_idx:>6}", end="")
        b1s = []
        mps = []
        for cat, r in all_results.items():
            lr = [l for l in r['layers'] if l['layer'] == layer_idx][0]
            print(f"  {lr['beta1']:>8}", end="")
            b1s.append(lr['beta1'])
            mps.append(lr['max_pers'])
        print(f"  {np.mean(b1s):>7.1f}  {np.mean(mps):>7.2f}")

    # Peak layer per prompt
    print(f"\n  Peak layers (highest max_persistence):")
    for cat, r in all_results.items():
        peak = max(r['layers'], key=lambda l: l['max_pers'])
        print(f"    {cat}: layer {peak['layer']} (β₁={peak['beta1']}, mp={peak['max_pers']:.4f})")

    # Layer 15 hypothesis
    print(f"\n  Layer 15 hypothesis test:")
    for cat, r in all_results.items():
        l15 = [l for l in r['layers'] if l['layer'] == 15]
        if l15:
            l15 = l15[0]
            peak = max(r['layers'], key=lambda l: l['max_pers'])
            is_peak = "✅ PEAK" if peak['layer'] == 15 else f"❌ peak={peak['layer']}"
            print(f"    {cat}: layer 15 β₁={l15['beta1']}, mp={l15['max_pers']:.4f} — {is_peak}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)
    print(f"\n  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
