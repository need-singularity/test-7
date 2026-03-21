"""
Experiment C: 쌍곡 Ricci Flow — Exp-A + Exp-B 결합

핵심 아이디어:
  쌍곡 공간은 본래 음의 곡률 → Ricci flow 동작이 유클리드와 근본적으로 다름.
  쌍곡 거리 기반 k-NN 그래프에서 ORC flow를 실행하여
  유클리드 Ricci flow(Exp-A)와 수렴 속도/결과 비교.

의존성:
  - Exp-A: Ricci flow 구현 (ORC, flow step)
  - Exp-B: 쌍곡 거리 계산 (Poincaré ball)
"""

import sys
import json
import time
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_model, extract_embeddings, compute_full_topology,
    compute_topology_from_distance_matrix,
    emergence_score, OUTPUT_DIR, PROMPTS_SUBSET, numpy_converter,
)
from expA_ricci_flow import (
    compute_all_curvatures, ricci_flow_step,
    graph_to_distance_matrix, direction_preservation,
)
from expB_hyperbolic_ph import (
    project_to_poincare_ball, poincare_distance_matrix,
    delta_hyperbolicity, hierarchy_score,
)


def build_knn_graph_from_dm(dm, k=10):
    """거리 행렬에서 k-NN 그래프 구축."""
    n = dm.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        neighbors = np.argsort(dm[i])[1:k+1]
        for j in neighbors:
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(dm[i, j]))

    return G


def run_combined_experiment(llm, category, prompt, max_iter=50, epsilon=0.3, k=10):
    """단일 프롬프트: 유클리드 Ricci vs 쌍곡 Ricci 비교."""
    print(f"\n{'='*70}")
    print(f"[Exp-C] [{category}] \"{prompt}\"")
    print(f"{'='*70}")

    # 1. Embeddings → PCA
    t0 = time.time()
    embeddings = extract_embeddings(llm, prompt)
    n, d = embeddings.shape
    pca_dim = min(50, n - 1, d)
    reduced = PCA(n_components=pca_dim).fit_transform(embeddings)
    print(f"  Embeddings: {n}x{d} → PCA {pca_dim}-dim ({time.time()-t0:.1f}s)")

    # 2. 원본 위상
    topo_orig = compute_full_topology(embeddings)
    print(f"  Original: β₁={topo_orig['beta1']}, max_pers={topo_orig['max_persistence']:.4f}")

    if topo_orig['beta1'] == 0:
        print("  [NO WALLS] Skipping.")
        return None

    # 3. δ-hyperbolicity
    euclidean_dm = squareform(pdist(reduced))
    delta = delta_hyperbolicity(euclidean_dm)
    h_score = hierarchy_score(delta)
    print(f"  δ-hyperbolicity: δ={delta:.4f}, hierarchy={h_score:.4f}")

    # 4. 쌍곡 거리 행렬 + 쌍곡 PH baseline
    poincare_points = project_to_poincare_ball(reduced)
    hyper_dm = poincare_distance_matrix(poincare_points)
    topo_hyper_orig = compute_topology_from_distance_matrix(hyper_dm)
    print(f"  Hyperbolic baseline: β₁={topo_hyper_orig['beta1']}, max_pers={topo_hyper_orig['max_persistence']:.4f}")

    # ── 유클리드 Ricci Flow ──
    print(f"\n  --- Euclidean Ricci Flow ---")
    G_euclid = build_knn_graph_from_dm(euclidean_dm, k=k)
    euclid_history = []

    for it in range(max_iter):
        curvatures = compute_all_curvatures(G_euclid, alpha=0.5)
        dm_current = graph_to_distance_matrix(G_euclid)
        topo_current = compute_topology_from_distance_matrix(dm_current)

        mean_k = float(np.mean(list(curvatures.values()))) if curvatures else 0.0
        euclid_history.append({
            'iteration': it,
            'beta1': topo_current['beta1'],
            'max_persistence': topo_current['max_persistence'],
            'mean_curvature': mean_k,
        })

        if it % 10 == 0:
            print(f"    iter={it:3d}  β₁={topo_current['beta1']}  max_pers={topo_current['max_persistence']:.4f}  κ={mean_k:+.4f}")

        if topo_current['beta1'] == 0:
            print(f"    ★ β₁=0 at iter {it}")
            break

        G_euclid = ricci_flow_step(G_euclid, curvatures, epsilon=epsilon)

    dm_euclid_final = graph_to_distance_matrix(G_euclid)
    topo_euclid_final = compute_topology_from_distance_matrix(dm_euclid_final)

    # ── 쌍곡 Ricci Flow ──
    print(f"\n  --- Hyperbolic Ricci Flow ---")
    G_hyper = build_knn_graph_from_dm(hyper_dm, k=k)
    hyper_history = []

    for it in range(max_iter):
        curvatures = compute_all_curvatures(G_hyper, alpha=0.5)
        dm_current = graph_to_distance_matrix(G_hyper)
        topo_current = compute_topology_from_distance_matrix(dm_current)

        mean_k = float(np.mean(list(curvatures.values()))) if curvatures else 0.0
        hyper_history.append({
            'iteration': it,
            'beta1': topo_current['beta1'],
            'max_persistence': topo_current['max_persistence'],
            'mean_curvature': mean_k,
        })

        if it % 10 == 0:
            print(f"    iter={it:3d}  β₁={topo_current['beta1']}  max_pers={topo_current['max_persistence']:.4f}  κ={mean_k:+.4f}")

        if topo_current['beta1'] == 0:
            print(f"    ★ β₁=0 at iter {it}")
            break

        G_hyper = ricci_flow_step(G_hyper, curvatures, epsilon=epsilon)

    dm_hyper_final = graph_to_distance_matrix(G_hyper)
    topo_hyper_final = compute_topology_from_distance_matrix(dm_hyper_final)

    # ── 비교 ──
    euclid_converged = euclid_history[-1]['beta1'] if euclid_history else topo_orig['beta1']
    hyper_converged = hyper_history[-1]['beta1'] if hyper_history else topo_hyper_orig['beta1']
    euclid_iters = len(euclid_history)
    hyper_iters = len(hyper_history)

    # β₁=0 도달 iteration 찾기
    euclid_zero_iter = next((h['iteration'] for h in euclid_history if h['beta1'] == 0), None)
    hyper_zero_iter = next((h['iteration'] for h in hyper_history if h['beta1'] == 0), None)

    print(f"\n  ── Comparison ──")
    print(f"  {'':>25} {'Euclidean':>12} {'Hyperbolic':>12}")
    print(f"  {'Original β₁':>25} {topo_orig['beta1']:>12} {topo_hyper_orig['beta1']:>12}")
    print(f"  {'Final β₁':>25} {topo_euclid_final['beta1']:>12} {topo_hyper_final['beta1']:>12}")
    print(f"  {'β₁=0 iter':>25} {str(euclid_zero_iter):>12} {str(hyper_zero_iter):>12}")
    print(f"  {'Total iters':>25} {euclid_iters:>12} {hyper_iters:>12}")
    print(f"  {'Final max_pers':>25} {topo_euclid_final['max_persistence']:>12.4f} {topo_hyper_final['max_persistence']:>12.4f}")

    if hyper_zero_iter is not None and euclid_zero_iter is not None:
        if hyper_zero_iter < euclid_zero_iter:
            print(f"    → 쌍곡 Ricci가 {euclid_zero_iter - hyper_zero_iter}회 더 빨리 수렴!")
        elif euclid_zero_iter < hyper_zero_iter:
            print(f"    → 유클리드 Ricci가 {hyper_zero_iter - euclid_zero_iter}회 더 빨리 수렴.")
        else:
            print(f"    → 동일 속도로 수렴.")

    # Emergence scores
    score_euclid = emergence_score(topo_orig, topo_euclid_final)
    score_hyper = emergence_score(topo_orig, topo_hyper_final)

    return {
        'category': category,
        'prompt': prompt,
        'delta_hyperbolicity': delta,
        'hierarchy_score': h_score,
        'original': {
            'euclidean_beta1': topo_orig['beta1'],
            'hyperbolic_beta1': topo_hyper_orig['beta1'],
            'euclidean_max_pers': topo_orig['max_persistence'],
            'hyperbolic_max_pers': topo_hyper_orig['max_persistence'],
        },
        'euclidean_ricci': {
            'final_beta1': topo_euclid_final['beta1'],
            'final_max_pers': topo_euclid_final['max_persistence'],
            'zero_iter': euclid_zero_iter,
            'total_iters': euclid_iters,
            'passage_score': score_euclid['passage_score'],
            'history': euclid_history,
        },
        'hyperbolic_ricci': {
            'final_beta1': topo_hyper_final['beta1'],
            'final_max_pers': topo_hyper_final['max_persistence'],
            'zero_iter': hyper_zero_iter,
            'total_iters': hyper_iters,
            'passage_score': score_hyper['passage_score'],
            'history': hyper_history,
        },
        'parameters': {'k': k, 'epsilon': epsilon, 'max_iter': max_iter},
    }


def run():
    llm = load_model()
    all_results = {}

    for category, prompt in PROMPTS_SUBSET:
        result = run_combined_experiment(llm, category, prompt)
        if result:
            all_results[category] = result

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "expC_combined_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=numpy_converter)

    # 요약
    print(f"\n\n{'='*70}")
    print("EXP-C: EUCLIDEAN vs HYPERBOLIC RICCI FLOW")
    print(f"{'='*70}\n")

    print(f"{'Category':<12} {'δ':>6} "
          f"{'E.β₁':>5}→{'E.fin':>5} {'E.iter':>6} {'E.score':>7} "
          f"{'H.β₁':>5}→{'H.fin':>5} {'H.iter':>6} {'H.score':>7} "
          f"{'Winner':>10}")
    print("-" * 95)

    for cat, data in all_results.items():
        e = data['euclidean_ricci']
        h = data['hyperbolic_ricci']
        o = data['original']

        e_iter_str = str(e['zero_iter']) if e['zero_iter'] is not None else f">{e['total_iters']}"
        h_iter_str = str(h['zero_iter']) if h['zero_iter'] is not None else f">{h['total_iters']}"

        if h['zero_iter'] is not None and (e['zero_iter'] is None or h['zero_iter'] < e['zero_iter']):
            winner = "HYPER"
        elif e['zero_iter'] is not None and (h['zero_iter'] is None or e['zero_iter'] < h['zero_iter']):
            winner = "EUCLID"
        elif e['passage_score'] > h['passage_score']:
            winner = "euclid"
        elif h['passage_score'] > e['passage_score']:
            winner = "hyper"
        else:
            winner = "tie"

        print(f"{cat:<12} {data['delta_hyperbolicity']:>6.2f} "
              f"{o['euclidean_beta1']:>5}→{e['final_beta1']:>5} {e_iter_str:>6} {e['passage_score']:>7.4f} "
              f"{o['hyperbolic_beta1']:>5}→{h['final_beta1']:>5} {h_iter_str:>6} {h['passage_score']:>7.4f} "
              f"{winner:>10}")

    print(f"\nSaved to {OUTPUT_DIR / 'expC_combined_results.json'}")


if __name__ == "__main__":
    run()
