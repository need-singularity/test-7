"""
Experiment B: 쌍곡선 임베딩 PH — Poincaré Disk 거리

핵심 아이디어:
  LLM 잠재 공간은 계층적 구조 → 쌍곡 거리가 유클리드보다 정확할 수 있음.
  동일 point cloud에서 유클리드 vs 쌍곡 거리로 β₁이 어떻게 달라지는지 비교.

파이프라인:
  1. extract_embeddings → PCA 50-dim point cloud
  2. δ-hyperbolicity 측정 (4-point condition)
  3. Poincaré ball로 사영 (exponential map)
  4. 쌍곡 거리 행렬 계산
  5. 유클리드 vs 쌍곡 Ripser PH 비교
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_model, extract_embeddings, compute_full_topology,
    compute_topology_from_distance_matrix,
    OUTPUT_DIR, PROMPTS_STANDARD, numpy_converter,
)


# ── δ-Hyperbolicity ───────────────────────────────────

def delta_hyperbolicity(dm):
    """Gromov δ-hyperbolicity (4-point condition).

    O(n⁴) — n≈40이면 ~2.5M quadruple, <1초.
    δ=0 → 완벽한 트리 (쌍곡적), δ 클수록 비트리적.
    """
    n = dm.shape[0]
    delta = 0.0

    for x in range(n):
        for y in range(x + 1, n):
            for z in range(y + 1, n):
                for w in range(z + 1, n):
                    s1 = dm[x, y] + dm[z, w]
                    s2 = dm[x, z] + dm[y, w]
                    s3 = dm[x, w] + dm[y, z]

                    sums = sorted([s1, s2, s3], reverse=True)
                    d = (sums[0] - sums[1]) / 2.0
                    delta = max(delta, d)

    return delta


def hierarchy_score(delta):
    """1/(1+δ). 1.0 = 완벽한 트리, 0에 가까울수록 비트리적."""
    return 1.0 / (1.0 + delta)


# ── Poincaré Ball 사영 ─────────────────────────────────

def project_to_poincare_ball(points, curvature=1.0, eps=1e-5):
    """유클리드 점들을 Poincaré ball 내부로 사영.

    exp_0(v) = tanh(||v|| * sqrt(c) / 2) * v / (||v|| * sqrt(c))
    base = origin (0벡터)
    """
    sqrt_c = np.sqrt(curvature)
    projected = np.zeros_like(points)

    for i, v in enumerate(points):
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            projected[i] = v
            continue

        # exponential map from origin
        scale = np.tanh(norm * sqrt_c / 2.0) / (norm * sqrt_c)
        projected[i] = v * scale

        # 경계 안전: ||x|| < 1 - eps
        p_norm = np.linalg.norm(projected[i])
        max_norm = 1.0 - eps
        if p_norm >= max_norm:
            projected[i] = projected[i] * (max_norm / p_norm)

    return projected


def poincare_distance(x, y, curvature=1.0, eps=1e-7):
    """Poincaré ball 거리.

    d_H(x,y) = (1/sqrt(c)) * arccosh(1 + 2c * ||x-y||² / ((1-c||x||²)(1-c||y||²)))
    """
    c = curvature
    diff_sq = np.sum((x - y) ** 2)
    x_sq = np.sum(x ** 2)
    y_sq = np.sum(y ** 2)

    denom = (1.0 - c * x_sq) * (1.0 - c * y_sq)
    denom = max(denom, eps)  # 수치 안정

    arg = 1.0 + 2.0 * c * diff_sq / denom
    arg = max(arg, 1.0 + eps)  # arccosh 정의역

    return np.arccosh(arg) / np.sqrt(c)


def poincare_distance_matrix(points, curvature=1.0):
    """Poincaré ball 거리 행렬."""
    n = len(points)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = poincare_distance(points[i], points[j], curvature)
            dm[i, j] = d
            dm[j, i] = d
    return dm


# ── Main Experiment ────────────────────────────────────

def run_hyperbolic_experiment(llm, category, prompt, curvature=1.0):
    """단일 프롬프트: 유클리드 vs 쌍곡 PH 비교."""
    print(f"\n{'='*70}")
    print(f"[Exp-B] [{category}] \"{prompt}\"")
    print(f"{'='*70}")

    # 1. Embeddings → PCA
    t0 = time.time()
    embeddings = extract_embeddings(llm, prompt)
    n, d = embeddings.shape
    print(f"  Embeddings: {n}x{d} ({time.time()-t0:.1f}s)")

    pca_dim = min(50, n - 1, d)
    reduced = PCA(n_components=pca_dim).fit_transform(embeddings)
    print(f"  PCA: {d}-dim → {pca_dim}-dim")

    # 2. 유클리드 PH (baseline)
    euclidean_dm = squareform(pdist(reduced))
    topo_euclid = compute_full_topology(embeddings)
    print(f"  Euclidean: β₁={topo_euclid['beta1']}, max_pers={topo_euclid['max_persistence']:.4f}")

    # 3. δ-hyperbolicity
    t0 = time.time()
    delta = delta_hyperbolicity(euclidean_dm)
    h_score = hierarchy_score(delta)
    t_delta = time.time() - t0
    print(f"  δ-hyperbolicity: δ={delta:.4f}, hierarchy_score={h_score:.4f} ({t_delta:.1f}s)")

    if h_score > 0.5:
        print(f"    → 공간이 쌍곡적 (트리에 가까움). 쌍곡 PH가 유효할 가능성 높음.")
    else:
        print(f"    → 공간이 비쌍곡적. 쌍곡 PH가 부적합할 수 있음.")

    # 4. Poincaré ball 사영
    poincare_points = project_to_poincare_ball(reduced, curvature=curvature)
    norms = np.linalg.norm(poincare_points, axis=1)
    print(f"  Poincaré ball: mean_norm={norms.mean():.4f}, max_norm={norms.max():.4f}")

    # 5. 쌍곡 거리 행렬
    t0 = time.time()
    hyper_dm = poincare_distance_matrix(poincare_points, curvature=curvature)
    t_dm = time.time() - t0
    print(f"  Hyperbolic distance matrix: ({t_dm:.1f}s)")
    print(f"    mean_dist={hyper_dm[hyper_dm > 0].mean():.4f}, max_dist={hyper_dm.max():.4f}")

    # 6. 쌍곡 PH
    topo_hyper = compute_topology_from_distance_matrix(hyper_dm)
    print(f"  Hyperbolic: β₁={topo_hyper['beta1']}, max_pers={topo_hyper['max_persistence']:.4f}")

    # 7. 비교
    print(f"\n  ── Comparison ──")
    print(f"  {'':>20} {'Euclidean':>12} {'Hyperbolic':>12} {'Delta':>8}")
    print(f"  {'β₁':>20} {topo_euclid['beta1']:>12} {topo_hyper['beta1']:>12} {topo_hyper['beta1'] - topo_euclid['beta1']:>+8}")
    print(f"  {'max_persistence':>20} {topo_euclid['max_persistence']:>12.4f} {topo_hyper['max_persistence']:>12.4f} {topo_hyper['max_persistence'] - topo_euclid['max_persistence']:>+8.4f}")
    print(f"  {'total_persistence':>20} {topo_euclid['total_persistence']:>12.4f} {topo_hyper['total_persistence']:>12.4f} {topo_hyper['total_persistence'] - topo_euclid['total_persistence']:>+8.4f}")

    # 쌍곡 벽 상세
    if topo_hyper['walls']:
        print(f"\n  Hyperbolic walls:")
        for i, w in enumerate(topo_hyper['walls'][:5]):
            print(f"    wall {i}: pers={w['persistence']:.4f}, verts={w['vertex_indices'][:6]}{'...' if len(w['vertex_indices']) > 6 else ''}")

    return {
        'category': category,
        'prompt': prompt,
        'n_points': n,
        'pca_dim': pca_dim,
        'curvature': curvature,
        'delta_hyperbolicity': delta,
        'hierarchy_score': h_score,
        'poincare_mean_norm': float(norms.mean()),
        'poincare_max_norm': float(norms.max()),
        'euclidean': {
            'beta1': topo_euclid['beta1'],
            'max_persistence': topo_euclid['max_persistence'],
            'total_persistence': topo_euclid['total_persistence'],
            'n_walls': len(topo_euclid['walls']),
        },
        'hyperbolic': {
            'beta1': topo_hyper['beta1'],
            'max_persistence': topo_hyper['max_persistence'],
            'total_persistence': topo_hyper['total_persistence'],
            'n_walls': len(topo_hyper['walls']),
            'walls': [{k: v for k, v in w.items() if k != 'idx'} for w in topo_hyper['walls'][:10]],
        },
        'beta1_delta': topo_hyper['beta1'] - topo_euclid['beta1'],
        'max_pers_delta': topo_hyper['max_persistence'] - topo_euclid['max_persistence'],
    }


def run():
    llm = load_model()
    all_results = {}

    for category, prompt in PROMPTS_STANDARD:
        result = run_hyperbolic_experiment(llm, category, prompt)
        all_results[category] = result

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "expB_hyperbolic_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=numpy_converter)

    # 요약
    print(f"\n\n{'='*70}")
    print("EXP-B: EUCLIDEAN vs HYPERBOLIC PH COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Category':<12} {'δ':>6} {'H.Score':>8} "
          f"{'E.β₁':>5} {'H.β₁':>5} {'Δβ₁':>5} "
          f"{'E.MaxP':>7} {'H.MaxP':>7} {'ΔMaxP':>7}")
    print("-" * 75)

    for cat, data in all_results.items():
        e = data['euclidean']
        h = data['hyperbolic']
        print(f"{cat:<12} {data['delta_hyperbolicity']:>6.2f} {data['hierarchy_score']:>8.4f} "
              f"{e['beta1']:>5} {h['beta1']:>5} {data['beta1_delta']:>+5} "
              f"{e['max_persistence']:>7.3f} {h['max_persistence']:>7.3f} {data['max_pers_delta']:>+7.3f}")

    # 해석
    print(f"\n  Interpretation:")
    avg_delta = np.mean([d['delta_hyperbolicity'] for d in all_results.values()])
    avg_h = np.mean([d['hierarchy_score'] for d in all_results.values()])
    print(f"    Average δ-hyperbolicity: {avg_delta:.4f}")
    print(f"    Average hierarchy score: {avg_h:.4f}")

    if avg_h > 0.5:
        print(f"    → 잠재 공간이 쌍곡적 경향. 쌍곡 PH가 더 정확한 벽 감지일 가능성.")
    else:
        print(f"    → 잠재 공간이 비쌍곡적. 유클리드 PH가 더 적합할 수 있음.")

    beta1_diff = [d['beta1_delta'] for d in all_results.values()]
    if any(d != 0 for d in beta1_diff):
        print(f"    → β₁ 차이 발견: 거리 메트릭이 위상 감지에 영향을 미침!")
    else:
        print(f"    → β₁ 동일: 거리 메트릭에 무관하게 같은 위상 감지.")

    print(f"\nSaved to {OUTPUT_DIR / 'expB_hyperbolic_results.json'}")


if __name__ == "__main__":
    run()
