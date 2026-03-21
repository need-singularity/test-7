"""
Experiment A: 푸앵카레 추측 기반 Ricci Flow 벽 통과

핵심 아이디어:
  β₁ hole이 있다 = 단일 연결(simply connected)이 아니다
  Ricci flow로 hole을 수축 → β₁=0 → S³ 복원 (단일 연결 조건)
  Phase 3b의 radial perturbation 대비 "곡률 기반 자동 조정"

파이프라인:
  1. extract_embeddings → PCA 50-dim point cloud
  2. k-NN 그래프 구축
  3. Ollivier-Ricci curvature 계산 (각 edge)
  4. Ricci flow iteration: edge weight 조정
  5. MDS로 flow된 거리 행렬 → 새 좌표 복원
  6. 매 iteration마다 β₁ 측정
  7. 방향 보존 검증
"""

import sys
import json
import time
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_model, extract_embeddings, compute_full_topology,
    compute_topology_from_distance_matrix,
    emergence_score, OUTPUT_DIR, PROMPTS_SUBSET, numpy_converter,
)


# ── Ollivier-Ricci Curvature ──────────────────────────

def build_knn_graph(points, k=10):
    """k-NN 그래프 구축. edge weight = 유클리드 거리."""
    dm = squareform(pdist(points))
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        neighbors = np.argsort(dm[i])[1:k+1]  # 자기 자신 제외
        for j in neighbors:
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(dm[i, j]))

    return G


def node_distribution(G, node, alpha=0.5):
    """노드의 확률 분포: alpha를 자기 자신에, (1-alpha)를 이웃에 균등 분배."""
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return {node: 1.0}

    dist = {node: alpha}
    share = (1.0 - alpha) / len(neighbors)
    for nb in neighbors:
        dist[nb] = share
    return dist


def wasserstein_1d(mu, nu, cost_matrix, nodes):
    """1-Wasserstein distance via linear programming.

    mu, nu: dict {node: probability}
    cost_matrix: pairwise cost between nodes
    nodes: list of node indices
    """
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    mu_vec = np.zeros(n)
    nu_vec = np.zeros(n)
    for node, prob in mu.items():
        if node in node_to_idx:
            mu_vec[node_to_idx[node]] = prob
    for node, prob in nu.items():
        if node in node_to_idx:
            nu_vec[node_to_idx[node]] = prob

    # LP: minimize sum c_ij * f_ij
    # subject to: sum_j f_ij = mu_i, sum_i f_ij = nu_j, f_ij >= 0
    c = cost_matrix[np.ix_(
        [node_to_idx.get(n, 0) for n in nodes],
        [node_to_idx.get(n, 0) for n in nodes]
    )].flatten()

    # Equality constraints: flow conservation
    n_vars = n * n
    A_eq = np.zeros((2 * n, n_vars))

    # Row sums = mu
    for i in range(n):
        A_eq[i, i*n:(i+1)*n] = 1.0

    # Column sums = nu
    for j in range(n):
        for i in range(n):
            A_eq[n + j, i*n + j] = 1.0

    b_eq = np.concatenate([mu_vec, nu_vec])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    if result.success:
        return result.fun
    return 0.0


def ollivier_ricci_curvature(G, edge, alpha=0.5):
    """Ollivier-Ricci curvature for a single edge.

    κ(x,y) = 1 - W₁(μ_x, μ_y) / d(x,y)
    """
    u, v = edge
    d_uv = G[u][v]['weight']
    if d_uv < 1e-10:
        return 0.0

    mu = node_distribution(G, u, alpha)
    nu = node_distribution(G, v, alpha)

    # 관련 노드들만 추출
    all_nodes = sorted(set(mu.keys()) | set(nu.keys()))

    # 비용 행렬: 그래프 최단 경로
    cost = np.zeros((len(all_nodes), len(all_nodes)))
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    for i, ni in enumerate(all_nodes):
        for j, nj in enumerate(all_nodes):
            if ni in lengths and nj in lengths[ni]:
                cost[i, j] = lengths[ni][nj]
            else:
                cost[i, j] = 1e6  # unreachable

    w1 = wasserstein_1d(mu, nu, cost, all_nodes)
    kappa = 1.0 - w1 / d_uv
    return kappa


def compute_all_curvatures(G, alpha=0.5):
    """그래프의 모든 edge에 대한 ORC 계산."""
    curvatures = {}
    for u, v in G.edges():
        kappa = ollivier_ricci_curvature(G, (u, v), alpha)
        curvatures[(u, v)] = kappa
    return curvatures


# ── Ricci Flow ─────────────────────────────────────────

def ricci_flow_step(G, curvatures, epsilon=0.1):
    """단일 Ricci flow step: w(e) -= ε * κ(e) * w(e)

    양의 곡률(κ>0) → weight 감소 (수축)
    음의 곡률(κ<0) → weight 증가 (확장)
    """
    G_new = G.copy()
    for (u, v), kappa in curvatures.items():
        old_w = G[u][v]['weight']
        new_w = old_w - epsilon * kappa * old_w
        new_w = max(new_w, 1e-6)  # 양수 유지
        G_new[u][v]['weight'] = new_w
    return G_new


def graph_to_distance_matrix(G):
    """그래프의 최단 경로 거리 행렬."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    dm = np.zeros((n, n))
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if ni in lengths and nj in lengths[ni]:
                dm[i, j] = lengths[ni][nj]
            else:
                dm[i, j] = 1e6
    return dm


def reconstruct_points_mds(dm, n_components=50):
    """MDS로 거리 행렬에서 좌표 복원."""
    mds = MDS(n_components=min(n_components, len(dm) - 1),
              dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    return mds.fit_transform(dm)


def direction_preservation(points_before, points_after):
    """방향 보존 측정: pairwise cosine similarity의 평균."""
    n = len(points_before)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            v_before = points_before[j] - points_before[i]
            v_after = points_after[j] - points_after[i]
            nb = np.linalg.norm(v_before)
            na = np.linalg.norm(v_after)
            if nb > 1e-10 and na > 1e-10:
                sim = 1.0 - cosine(v_before, v_after)
                similarities.append(sim)
    return float(np.mean(similarities)) if similarities else 0.0


# ── Main Experiment ────────────────────────────────────

def run_ricci_experiment(llm, category, prompt, max_iter=50, epsilon=0.3, k=10):
    """단일 프롬프트에 대한 Ricci flow 실험."""
    print(f"\n{'='*70}")
    print(f"[Exp-A] [{category}] \"{prompt}\"")
    print(f"{'='*70}")

    # 1. Embeddings
    t0 = time.time()
    embeddings = extract_embeddings(llm, prompt)
    n, d = embeddings.shape
    print(f"  Embeddings: {n}x{d} ({time.time()-t0:.1f}s)")

    # PCA → 50-dim
    pca_dim = min(50, n - 1, d)
    reduced = PCA(n_components=pca_dim).fit_transform(embeddings)
    print(f"  PCA: {d}-dim → {pca_dim}-dim")

    # 2. 원본 위상
    topo_orig = compute_full_topology(embeddings)
    print(f"  Original: β₁={topo_orig['beta1']}, max_pers={topo_orig['max_persistence']:.4f}")

    if topo_orig['beta1'] == 0:
        print("  [NO WALLS] Skipping.")
        return None

    # 3. k-NN 그래프 구축
    G = build_knn_graph(reduced, k=k)
    print(f"  k-NN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 4. Ricci flow iterations
    history = []
    best_beta1 = topo_orig['beta1']
    best_iter = 0
    points_original = reduced.copy()

    for it in range(max_iter):
        # Curvature 계산
        curvatures = compute_all_curvatures(G, alpha=0.5)

        # 현재 상태 기록
        dm = graph_to_distance_matrix(G)
        topo_current = compute_topology_from_distance_matrix(dm)

        mean_kappa = float(np.mean(list(curvatures.values()))) if curvatures else 0.0
        min_kappa = float(np.min(list(curvatures.values()))) if curvatures else 0.0
        max_kappa = float(np.max(list(curvatures.values()))) if curvatures else 0.0

        entry = {
            'iteration': it,
            'beta1': topo_current['beta1'],
            'max_persistence': topo_current['max_persistence'],
            'mean_curvature': mean_kappa,
            'min_curvature': min_kappa,
            'max_curvature': max_kappa,
        }
        history.append(entry)

        if it % 5 == 0 or topo_current['beta1'] < best_beta1:
            print(f"    iter={it:3d}  β₁={topo_current['beta1']}  max_pers={topo_current['max_persistence']:.4f}  "
                  f"κ_mean={mean_kappa:+.4f}  κ_range=[{min_kappa:.3f}, {max_kappa:.3f}]")

        if topo_current['beta1'] < best_beta1:
            best_beta1 = topo_current['beta1']
            best_iter = it

        # β₁=0 달성 → 조기 종료
        if topo_current['beta1'] == 0:
            print(f"    ★ β₁=0 at iteration {it}!")
            break

        # Flow step
        G = ricci_flow_step(G, curvatures, epsilon=epsilon)

    # 5. 최종 상태
    dm_final = graph_to_distance_matrix(G)
    topo_final = compute_topology_from_distance_matrix(dm_final)
    points_final = reconstruct_points_mds(dm_final, n_components=pca_dim)

    # 6. 방향 보존
    dir_pres = direction_preservation(points_original, points_final)
    print(f"\n  Final: β₁={topo_final['beta1']}, max_pers={topo_final['max_persistence']:.4f}")
    print(f"  Direction preservation: {dir_pres:.4f} {'✓' if dir_pres > 0.95 else '✗ (< 0.95)'}")
    print(f"  Best β₁={best_beta1} at iter={best_iter}")

    # 7. Phase 3b radial 비교용 emergence score
    score = emergence_score(topo_orig, topo_final)
    print(f"  Passage score: {score['passage_score']:.4f}")
    print(f"    wall_reduction={score['wall_reduction']:.4f}  pers_reduction={score['pers_reduction']:.4f}  stability={score['stability']:.4f}")

    return {
        'category': category,
        'prompt': prompt,
        'original_beta1': topo_orig['beta1'],
        'original_max_pers': topo_orig['max_persistence'],
        'final_beta1': topo_final['beta1'],
        'final_max_pers': topo_final['max_persistence'],
        'best_beta1': best_beta1,
        'best_iteration': best_iter,
        'total_iterations': len(history),
        'direction_preservation': dir_pres,
        'passage_score': score['passage_score'],
        'emergence_score': score,
        'parameters': {'k': k, 'epsilon': epsilon, 'max_iter': max_iter},
        'history': history,
    }


def run():
    llm = load_model()
    all_results = {}

    for category, prompt in PROMPTS_SUBSET:
        result = run_ricci_experiment(llm, category, prompt)
        if result:
            all_results[category] = result

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "expA_ricci_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=numpy_converter)

    # 요약
    print(f"\n\n{'='*70}")
    print("EXP-A: RICCI FLOW WALL PASSAGE RESULTS")
    print(f"{'='*70}\n")

    print(f"{'Category':<12} {'β₁ orig':>8} {'β₁ final':>9} {'Best β₁':>8} {'Iter':>5} "
          f"{'Dir.Pres':>9} {'Score':>7}")
    print("-" * 65)

    for cat, data in all_results.items():
        print(f"{cat:<12} {data['original_beta1']:>8} {data['final_beta1']:>9} "
              f"{data['best_beta1']:>8} {data['best_iteration']:>5} "
              f"{data['direction_preservation']:>9.4f} {data['passage_score']:>7.4f}")

    # Radial 비교 (Phase 4 데이터)
    phase4_path = OUTPUT_DIR / "phase4_optimization_results.json"
    if phase4_path.exists():
        with open(phase4_path) as f:
            phase4 = json.load(f)
        print(f"\n  Comparison with Phase 3b/4 Radial:")
        for cat in all_results:
            if cat in phase4:
                p4 = phase4[cat]
                print(f"    {cat}: Radial β₁={p4['best_beta1']} (α={p4['best_alpha']:.1f})"
                      f" vs Ricci β₁={all_results[cat]['final_beta1']} (iter={all_results[cat]['best_iteration']})")

    print(f"\nSaved to {OUTPUT_DIR / 'expA_ricci_results.json'}")


if __name__ == "__main__":
    run()
