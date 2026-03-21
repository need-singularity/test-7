"""
실제 Llama 8B Hidden State에서 T4 + T5 검증
=============================================

합성 50-dim이 아닌 실제 4096-dim hidden state로:
  T4: Wall dims vs Random dims — selective 수축 효과 비교
  T5: PH artifact — noise vs real hidden state β₁/persistence
      + permutation test + null model
"""

import numpy as np
import time
import json
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.decomposition import PCA
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = Path("/Users/ghost/Dev/fire-in-the-hole/data/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
PERSISTENCE_FLOOR = 0.05
PCA_DIM = 50  # 프로젝트 표준

PROMPTS = [
    ("creative",  "A color that doesn't exist yet would look like"),
    ("factual",   "The capital of France is"),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
    ("boundary",  "The solution to the Riemann hypothesis involves"),
    ("creative2", "If mathematics were a living organism, its heartbeat would be"),
]


# ── Model & Embedding ──────────────────────────────────────────────────────

def load_model():
    from llama_cpp import Llama
    print("Loading Llama 8B GGUF...")
    t0 = time.time()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_gpu_layers=-1,
        embedding=True,
        verbose=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return llm


def extract_embeddings(llm, prompt):
    """프로젝트 phase3b와 동일한 방식으로 embedding 추출"""
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


# ── PH helpers ──────────────────────────────────────────────────────────────

def compute_ph(points, pca_dim=PCA_DIM, floor=PERSISTENCE_FLOOR):
    """PCA 축소 후 PH 계산 — 프로젝트 표준 파이프라인"""
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
            walls.append({
                'persistence': float(p),
                'vertex_indices': verts,
                'birth': float(b),
                'death': float(d_),
            })
    walls.sort(key=lambda w: w['persistence'], reverse=True)

    all_pers = sorted([float(d_ - b) for b, d_ in dgm1 if np.isfinite(d_)], reverse=True)

    return {
        'beta1': len(walls),
        'max_persistence': walls[0]['persistence'] if walls else 0.0,
        'total_persistence': sum(w['persistence'] for w in walls),
        'walls': walls,
        'all_pers': all_pers,
        'reduced': reduced,
    }


def get_wall_dims(walls, embeddings):
    """Wall에서 가장 기여도 높은 차원 식별 (프로젝트 phase2 방식 간소화)"""
    if not walls:
        return []

    # 모든 wall vertex에서 variance가 높은 원본 차원 = wall dim 후보
    all_verts = set()
    for w in walls:
        all_verts.update(w['vertex_indices'])
    all_verts = sorted(all_verts)

    if len(all_verts) < 3:
        return []

    wall_points = embeddings[all_verts]
    variances = np.var(wall_points, axis=0)

    # top-10 high-variance dims
    wall_dims = sorted(np.argsort(variances)[-10:].tolist())
    return wall_dims


# ── Contraction (4096-dim 원본 공간에서) ─────────────────────────────────

def contract_selective_4096(embeddings, dims, rate):
    out = embeddings.copy()
    center = embeddings[:, dims].mean(axis=0)
    out[:, dims] = embeddings[:, dims] - rate * (embeddings[:, dims] - center)
    return out


def contract_global_4096(embeddings, rate):
    center = embeddings.mean(axis=0)
    return embeddings - rate * (embeddings - center)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    llm = load_model()

    all_results = {}

    for cat, prompt in PROMPTS:
        print(f"\n{'='*70}")
        print(f"[{cat}] \"{prompt}\"")
        print(f"{'='*70}")

        embeddings = extract_embeddings(llm, prompt)
        n_pts, full_dim = embeddings.shape
        print(f"  Embeddings: {embeddings.shape}")

        # ── Baseline PH ────────────────────────────────────────────────
        ph_orig = compute_ph(embeddings)
        print(f"  β₁={ph_orig['beta1']}, max_pers={ph_orig['max_persistence']:.4f}, "
              f"total_pers={ph_orig['total_persistence']:.4f}")

        # ── Wall dims 식별 ─────────────────────────────────────────────
        wall_dims = get_wall_dims(ph_orig['walls'], embeddings)
        print(f"  Wall dims (top-10 variance): {wall_dims}")

        if not wall_dims:
            print("  [NO WALLS] Skipping.")
            continue

        # ══════════════════════════════════════════════════════════════
        # T4: WALL vs RANDOM vs GLOBAL — 12-step
        # ══════════════════════════════════════════════════════════════
        print(f"\n  ── T4: Wall vs Random vs Global (12 steps, rate=0.15)")

        n_steps = 12
        rate = 0.15
        k = len(wall_dims)

        # Wall selective
        pts_wall = embeddings.copy()
        wall_traj = [ph_orig['beta1']]
        wall_pers_traj = [ph_orig['max_persistence']]
        for step in range(n_steps):
            pts_wall = contract_selective_4096(pts_wall, wall_dims, rate)
            ph = compute_ph(pts_wall)
            wall_traj.append(ph['beta1'])
            wall_pers_traj.append(ph['max_persistence'])

        # Global
        pts_glob = embeddings.copy()
        glob_traj = [ph_orig['beta1']]
        glob_pers_traj = [ph_orig['max_persistence']]
        for step in range(n_steps):
            pts_glob = contract_global_4096(pts_glob, rate)
            ph = compute_ph(pts_glob)
            glob_traj.append(ph['beta1'])
            glob_pers_traj.append(ph['max_persistence'])

        # Random-k (10 sets)
        n_random = 10
        rand_finals_b1 = []
        rand_finals_mp = []
        rand_trajs = []
        for r_trial in range(n_random):
            rng = np.random.RandomState(3000 + r_trial)
            random_dims = sorted(rng.choice(full_dim, size=k, replace=False).tolist())
            pts_rand = embeddings.copy()
            r_traj = [ph_orig['beta1']]
            for step in range(n_steps):
                pts_rand = contract_selective_4096(pts_rand, random_dims, rate)
                ph = compute_ph(pts_rand)
                r_traj.append(ph['beta1'])
            rand_finals_b1.append(r_traj[-1])
            rand_finals_mp.append(compute_ph(pts_rand)['max_persistence'])
            rand_trajs.append(r_traj)

        rand_mean_traj = np.mean(rand_trajs, axis=0)

        print(f"  {'Step':>5}  {'Wall':>6}  {'Global':>8}  {'Rand(mean)':>11}")
        print(f"  {'-'*34}")
        for step in [0, 3, 6, 9, 12]:
            print(f"  {step:>5}  {wall_traj[step]:>6}  {glob_traj[step]:>8}  {rand_mean_traj[step]:>11.1f}")

        print(f"\n  Final β₁:  Wall={wall_traj[-1]}  Global={glob_traj[-1]}  "
              f"Random(mean)={np.mean(rand_finals_b1):.1f} (best={min(rand_finals_b1)})")
        print(f"  Final max_pers:  Wall={wall_pers_traj[-1]:.4f}  Global={glob_pers_traj[-1]:.4f}  "
              f"Random(mean)={np.mean(rand_finals_mp):.4f}")

        # ══════════════════════════════════════════════════════════════
        # T5: PH ARTIFACT on real hidden states
        # ══════════════════════════════════════════════════════════════
        print(f"\n  ── T5: PH Artifact — permutation test on real embeddings")

        # Permutation: row-shuffle (구조 파괴, marginal 보존)
        n_perm = 20
        perm_b1s = []
        perm_mps = []
        for p_trial in range(n_perm):
            rng_p = np.random.RandomState(4000 + p_trial)
            perm = embeddings.copy()
            for i in range(n_pts):
                rng_p.shuffle(perm[i])
            ph_perm = compute_ph(perm)
            perm_b1s.append(ph_perm['beta1'])
            perm_mps.append(ph_perm['max_persistence'])

        # Column-shuffle (점 간 관계 파괴)
        col_b1s = []
        col_mps = []
        for p_trial in range(n_perm):
            rng_p = np.random.RandomState(5000 + p_trial)
            col_shuf = embeddings.copy()
            for j in range(full_dim):
                rng_p.shuffle(col_shuf[:, j])
            ph_col = compute_ph(col_shuf)
            col_b1s.append(ph_col['beta1'])
            col_mps.append(ph_col['max_persistence'])

        # Matched-norm noise
        mn_b1s = []
        mn_mps = []
        real_norms = np.linalg.norm(embeddings, axis=1)
        for p_trial in range(n_perm):
            rng_p = np.random.RandomState(6000 + p_trial)
            noise = rng_p.randn(n_pts, full_dim)
            noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
            matched = noise / (noise_norms + 1e-10) * real_norms[:, None]
            ph_mn = compute_ph(matched)
            mn_b1s.append(ph_mn['beta1'])
            mn_mps.append(ph_mn['max_persistence'])

        print(f"\n  {'':>20}  {'β₁':>6}  {'max_pers':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Real (original)':>20}  {ph_orig['beta1']:>6}  {ph_orig['max_persistence']:>10.4f}")
        print(f"  {'Row-shuffle':>20}  {np.mean(perm_b1s):>6.1f}  {np.mean(perm_mps):>10.4f}")
        print(f"  {'Col-shuffle':>20}  {np.mean(col_b1s):>6.1f}  {np.mean(col_mps):>10.4f}")
        print(f"  {'Matched-norm noise':>20}  {np.mean(mn_b1s):>6.1f}  {np.mean(mn_mps):>10.4f}")

        # p-value: real β₁이 null 대비 극단인가?
        # Col-shuffle을 null로 사용 (marginal 보존)
        col_all = col_b1s
        pctile = np.mean([c <= ph_orig['beta1'] for c in col_all]) * 100
        print(f"\n  Null model (col-shuffle): real β₁={ph_orig['beta1']} is at {pctile:.0f}th percentile")

        # Max persistence null
        mp_pctile = np.mean([c <= ph_orig['max_persistence'] for c in col_mps]) * 100
        print(f"  Null model (col-shuffle): real max_pers={ph_orig['max_persistence']:.4f} is at {mp_pctile:.0f}th percentile")

        # 결과 저장
        all_results[cat] = {
            'prompt': prompt,
            'n_pts': n_pts,
            'full_dim': full_dim,
            'original': {
                'beta1': ph_orig['beta1'],
                'max_pers': ph_orig['max_persistence'],
                'total_pers': ph_orig['total_persistence'],
            },
            'wall_dims': wall_dims,
            't4': {
                'wall_final_b1': wall_traj[-1],
                'wall_final_mp': wall_pers_traj[-1],
                'global_final_b1': glob_traj[-1],
                'global_final_mp': glob_pers_traj[-1],
                'random_final_b1_mean': float(np.mean(rand_finals_b1)),
                'random_final_b1_best': min(rand_finals_b1),
                'random_final_mp_mean': float(np.mean(rand_finals_mp)),
                'wall_traj': wall_traj,
                'glob_traj': glob_traj,
                'rand_mean_traj': rand_mean_traj.tolist(),
            },
            't5': {
                'row_shuffle_b1': float(np.mean(perm_b1s)),
                'row_shuffle_mp': float(np.mean(perm_mps)),
                'col_shuffle_b1': float(np.mean(col_b1s)),
                'col_shuffle_mp': float(np.mean(col_mps)),
                'matched_noise_b1': float(np.mean(mn_b1s)),
                'matched_noise_mp': float(np.mean(mn_mps)),
                'null_b1_percentile': pctile,
                'null_mp_percentile': mp_pctile,
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY — REAL Llama 8B Hidden States")
    print("=" * 70)

    # T4 Summary
    print(f"\n── T4: Wall vs Random vs Global (final β₁ after 12 steps)")
    print(f"  {'Category':>12}  {'Original':>9}  {'Wall':>6}  {'Global':>8}  {'Rand(mean)':>11}  {'Rand(best)':>11}  {'Wall wins?':>11}")
    print(f"  {'-'*75}")
    wall_wins = 0
    total = 0
    for cat, r in all_results.items():
        t4 = r['t4']
        w = t4['wall_final_b1']
        g = t4['global_final_b1']
        rm = t4['random_final_b1_mean']
        rb = t4['random_final_b1_best']
        win = "YES" if w < rm else ("TIE" if w == rm else "NO")
        if w < rm:
            wall_wins += 1
        total += 1
        print(f"  {cat:>12}  {r['original']['beta1']:>9}  {w:>6}  {g:>8}  {rm:>11.1f}  {rb:>11}  {win:>11}")

    print(f"\n  Wall wins: {wall_wins}/{total}")

    # T5 Summary
    print(f"\n── T5: PH Artifact — Is β₁ distinguishable from null?")
    print(f"  {'Category':>12}  {'Real β₁':>8}  {'Null β₁':>8}  {'Real mp':>8}  {'Null mp':>8}  {'β₁ %ile':>8}  {'mp %ile':>8}  {'Significant?':>13}")
    print(f"  {'-'*82}")
    sig_count = 0
    for cat, r in all_results.items():
        t5 = r['t5']
        sig = "YES" if t5['null_mp_percentile'] >= 95 else "no"
        if t5['null_mp_percentile'] >= 95:
            sig_count += 1
        print(f"  {cat:>12}  {r['original']['beta1']:>8}  {t5['col_shuffle_b1']:>8.1f}  "
              f"{r['original']['max_pers']:>8.4f}  {t5['col_shuffle_mp']:>8.4f}  "
              f"{t5['null_b1_percentile']:>7.0f}%  {t5['null_mp_percentile']:>7.0f}%  {sig:>13}")

    print(f"\n  Max persistence significant (p<0.05): {sig_count}/{total}")

    # Save
    out_path = Path("/Users/ghost/Dev/fire-in-the-hole/data/real_model_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
