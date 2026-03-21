"""
E1: 3-Way Embedding Contraction — Baseline vs Global vs Selective
=================================================================

실제 Llama 8B hidden state에서:
  1. Baseline: 원본 embedding의 β₁, persistence
  2. Global: 전체 dims 수축 후 β₁, persistence
  3. Selective: wall dims만 수축 후 β₁, persistence
  4. 수축 강도별 (α=0.05~0.80) sweep
  5. 구조 보존 지표: cosine similarity, L2 distance, wall/non-wall signal

8개 프롬프트 × 3-way × 8 α values
"""

import numpy as np
import time
import json
from pathlib import Path
from scipy.spatial.distance import pdist, squareform, cosine
from scipy import stats
from sklearn.decomposition import PCA
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
OUTPUT_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/e1_2way_results.json")
PCA_DIM = 50
PERSISTENCE_FLOOR = 0.05

PROMPTS = [
    ("creative",   "A color that doesn't exist yet would look like"),
    ("creative2",  "If mathematics were a living organism, its heartbeat would be"),
    ("factual",    "The capital of France is"),
    ("factual2",   "Water boils at a temperature of"),
    ("reasoning",  "If all roses are flowers and some flowers fade quickly, then"),
    ("reasoning2", "What comes next in the sequence: 2, 6, 18, 54, ?"),
    ("boundary",   "The solution to the Riemann hypothesis involves"),
    ("boundary2",  "The mechanism by which consciousness emerges from neurons is"),
]

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.80]


def load_model():
    from llama_cpp import Llama
    print("Loading Llama 8B GGUF...")
    t0 = time.time()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=512, n_gpu_layers=-1,
                embedding=True, verbose=False)
    print(f"  Loaded in {time.time()-t0:.1f}s")
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


def compute_ph(points, pca_dim=PCA_DIM, floor=PERSISTENCE_FLOOR):
    n, d = points.shape
    dim = min(pca_dim, n - 1, d)
    reduced = PCA(n_components=dim).fit_transform(points) if dim >= 2 else points
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


def get_wall_dims(walls, embeddings, k=10):
    all_verts = set()
    for w in walls:
        all_verts.update(w['vertex_indices'])
    if len(all_verts) < 3:
        return []
    wall_points = embeddings[sorted(all_verts)]
    variances = np.var(wall_points, axis=0)
    return sorted(np.argsort(variances)[-k:].tolist())


def measure_structure(original, modified, wall_dims):
    non_wall = sorted(set(range(original.shape[1])) - set(wall_dims))

    # Per-point cosine similarity
    cos_sims = []
    for i in range(len(original)):
        cs = 1 - cosine(original[i], modified[i])
        cos_sims.append(cs)

    # L2 distance
    l2_dists = np.linalg.norm(original - modified, axis=1)

    # Wall / non-wall signal change
    wall_sig_orig = np.mean(pdist(original[:, wall_dims])) if wall_dims else 0
    wall_sig_mod = np.mean(pdist(modified[:, wall_dims])) if wall_dims else 0
    nw_sig_orig = np.mean(pdist(original[:, non_wall]))
    nw_sig_mod = np.mean(pdist(modified[:, non_wall]))

    return {
        'cosine_sim_mean': float(np.mean(cos_sims)),
        'cosine_sim_min': float(np.min(cos_sims)),
        'l2_dist_mean': float(np.mean(l2_dists)),
        'l2_dist_max': float(np.max(l2_dists)),
        'wall_signal_change': float((wall_sig_mod - wall_sig_orig) / max(wall_sig_orig, 1e-9)),
        'nonwall_signal_change': float((nw_sig_mod - nw_sig_orig) / max(nw_sig_orig, 1e-9)),
    }


def main():
    llm = load_model()
    all_results = {}

    for cat, prompt in PROMPTS:
        print(f"\n{'='*70}")
        print(f"[{cat}] \"{prompt}\"")
        print(f"{'='*70}")

        embeddings = extract_embeddings(llm, prompt)
        n_pts, full_dim = embeddings.shape
        print(f"  Shape: {embeddings.shape}")

        # Baseline
        ph_base = compute_ph(embeddings)
        wall_dims = get_wall_dims(ph_base['walls'], embeddings)
        print(f"  Baseline: β₁={ph_base['beta1']}, max_pers={ph_base['max_pers']:.4f}")
        print(f"  Wall dims: {wall_dims}")

        if not wall_dims:
            print("  [NO WALLS] Skip.")
            continue

        # ── Two strategies: Global and Selective ────────────────────
        strategies = {
            'global': {},
            'selective': {},
        }

        for strategy_name in ['global', 'selective']:
            sweep = []
            print(f"\n  ── {strategy_name.upper()} ──")
            print(f"  {'α':>6}  {'β₁':>4}  {'max_pers':>9}  {'Δβ₁':>5}  {'cos_sim':>8}  {'L2':>8}  {'wall_Δ':>8}  {'nw_Δ':>8}")
            print(f"  {'-'*62}")
            print(f"  {'0.00':>6}  {ph_base['beta1']:>4}  {ph_base['max_pers']:>9.4f}  {'—':>5}  {'1.000':>8}  {'0.000':>8}  {'0.0%':>8}  {'0.0%':>8}")

            for alpha in ALPHAS:
                modified = embeddings.copy()
                if strategy_name == 'selective':
                    center = embeddings[:, wall_dims].mean(axis=0)
                    modified[:, wall_dims] -= alpha * (embeddings[:, wall_dims] - center)
                else:  # global
                    center = embeddings.mean(axis=0)
                    modified -= alpha * (embeddings - center)

                ph_mod = compute_ph(modified)
                struct = measure_structure(embeddings, modified, wall_dims)
                delta_b1 = ph_mod['beta1'] - ph_base['beta1']

                entry = {
                    'alpha': alpha,
                    'beta1': ph_mod['beta1'],
                    'max_pers': ph_mod['max_pers'],
                    'total_pers': ph_mod['total_pers'],
                    'delta_beta1': delta_b1,
                    **struct,
                }
                sweep.append(entry)

                print(f"  {alpha:>6.2f}  {ph_mod['beta1']:>4}  {ph_mod['max_pers']:>9.4f}  {delta_b1:>+5}  "
                      f"{struct['cosine_sim_mean']:>8.4f}  {struct['l2_dist_mean']:>8.3f}  "
                      f"{struct['wall_signal_change']*100:>+7.1f}%  {struct['nonwall_signal_change']*100:>+7.1f}%")

            # Best α: most β₁ reduction with cosine_sim > 0.95
            safe_entries = [e for e in sweep if e['cosine_sim_mean'] > 0.95]
            if safe_entries:
                best = min(safe_entries, key=lambda e: (e['beta1'], -e['cosine_sim_mean']))
                print(f"  ★ Best safe α={best['alpha']}: β₁={best['beta1']} (Δ{best['delta_beta1']:+d}), "
                      f"cos_sim={best['cosine_sim_mean']:.4f}")
            else:
                best = min(sweep, key=lambda e: e['beta1'])
                print(f"  ★ Best α={best['alpha']}: β₁={best['beta1']} (no safe α found)")

            strategies[strategy_name] = {
                'sweep': sweep,
                'best_alpha': best['alpha'] if safe_entries else None,
                'best': best,
            }

        all_results[cat] = {
            'prompt': prompt,
            'n_pts': n_pts,
            'wall_dims': wall_dims,
            'baseline': {
                'beta1': ph_base['beta1'],
                'max_pers': ph_base['max_pers'],
                'total_pers': ph_base['total_pers'],
            },
            'global': strategies['global'],
            'selective': strategies['selective'],
        }

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("E1 SUMMARY — 3-Way: Baseline vs Global vs Selective")
    print("=" * 70)

    print(f"\n  {'Category':>12}  {'β₁':>4}  │ {'G β₁':>5}  {'G α':>5}  {'G cos':>6}  {'G nw_Δ':>7}  │ {'S β₁':>5}  {'S α':>5}  {'S cos':>6}  {'S nw_Δ':>7}")
    print(f"  {'-'*85}")
    for cat, r in all_results.items():
        gb = r['global']['best']
        sb = r['selective']['best']
        print(f"  {cat:>12}  {r['baseline']['beta1']:>4}  │ "
              f"{gb['beta1']:>5}  {gb['alpha']:>5.2f}  {gb['cosine_sim_mean']:>6.4f}  {gb['nonwall_signal_change']*100:>+6.1f}%  │ "
              f"{sb['beta1']:>5}  {sb['alpha']:>5.2f}  {sb['cosine_sim_mean']:>6.4f}  {sb['nonwall_signal_change']*100:>+6.1f}%")

    # Winner per category
    print(f"\n  {'Category':>12}  {'Winner':>10}  {'Reason':>40}")
    print(f"  {'-'*65}")
    for cat, r in all_results.items():
        gb = r['global']['best']
        sb = r['selective']['best']
        base_b1 = r['baseline']['beta1']
        if gb['beta1'] < sb['beta1']:
            winner = "Global"
            reason = f"β₁ {base_b1}→{gb['beta1']} vs {sb['beta1']}"
        elif sb['beta1'] < gb['beta1']:
            winner = "Selective"
            reason = f"β₁ {base_b1}→{sb['beta1']} vs {gb['beta1']}"
        elif abs(gb['nonwall_signal_change']) < abs(sb['nonwall_signal_change']):
            winner = "Global"
            reason = f"Same β₁={gb['beta1']}, less nw damage"
        elif abs(sb['nonwall_signal_change']) < abs(gb['nonwall_signal_change']):
            winner = "Selective"
            reason = f"Same β₁={sb['beta1']}, less nw damage"
        else:
            winner = "TIE"
            reason = f"Same β₁={gb['beta1']}, same damage"
        print(f"  {cat:>12}  {winner:>10}  {reason:>40}")

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
