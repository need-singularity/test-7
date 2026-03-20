"""
Phase 3b: 비균일 섭동 — 벽 근처 점만 선택적으로 이동

Phase 3 실패 원인: 균일 translation은 거리 행렬 불변 → 위상 불변.
수정: cycle 꼭짓점만 passage direction으로 이동 → 거리 변화 → 위상 변화.

이것이 실제 adapter가 해야 하는 일의 정확한 모델:
  "벽을 구성하는 토큰의 표현만 새 방향으로 밀어서 hole을 닫는다"
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"
PERSISTENCE_THRESHOLD = 0.01


def load_model():
    from llama_cpp import Llama
    print("Loading model...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_gpu_layers=-1,
        embedding=True,
        verbose=False,
    )
    print("Loaded.\n")
    return llm


def extract_embeddings(llm, prompt):
    points = []
    emb = llm.embed(prompt)
    if isinstance(emb[0], list):
        points.extend(emb)
    else:
        points.append(emb)

    tokens = llm.tokenize(prompt.encode('utf-8'))
    for i in range(1, len(tokens)):
        prefix = llm.detokenize(tokens[:i]).decode('utf-8', errors='replace')
        if prefix.strip():
            emb = llm.embed(prefix)
            points.append(emb[-1] if isinstance(emb[0], list) else emb)

    suffixes = [" and", " but", " however", " because", " which",
                " the", " a", " in", " of", " that",
                ".", "?", "!", " not", " never",
                " always", " perhaps", " certainly", " impossible", " undefined"]
    for suffix in suffixes:
        emb = llm.embed(prompt + suffix)
        points.append(emb[-1] if isinstance(emb[0], list) else emb)

    return np.array(points)


def full_topology(points):
    """PH 계산 + cocycle 반환"""
    from ripser import ripser

    n, d = points.shape
    pca_dim = min(50, n - 1, d)
    if pca_dim >= 2:
        pca = PCA(n_components=pca_dim)
        reduced = pca.fit_transform(points)
    else:
        reduced = points

    dm = squareform(pdist(reduced))
    result = ripser(dm, maxdim=1, distance_matrix=True, do_cocycles=True)
    return result, reduced


def topo_summary(result):
    dgm = result['dgms'][1]
    bars = [(float(b), float(d), float(d-b)) for b, d in dgm
            if np.isfinite(d) and d - b > PERSISTENCE_THRESHOLD]
    bars.sort(key=lambda x: x[2], reverse=True)
    beta1 = len(bars)
    max_pers = bars[0][2] if bars else 0.0
    total_pers = sum(b[2] for b in bars)
    return {'beta1': beta1, 'max_pers': max_pers, 'total_pers': total_pers, 'bars': bars[:5]}


def extract_wall_info(result, points_full, points_pca):
    """가장 persistent한 β₁ hole의 cycle 정보 추출"""
    dgm = result['dgms'][1]
    cocycles = result['cocycles'][1]

    best_idx = -1
    best_pers = 0
    for idx, (b, d) in enumerate(dgm):
        p = d - b
        if np.isfinite(d) and p > best_pers:
            best_pers = p
            best_idx = idx

    if best_idx < 0:
        return None

    cocycle = cocycles[best_idx]
    verts = sorted(set(int(r[0]) for r in cocycle) | set(int(r[1]) for r in cocycle))

    if len(verts) < 3:
        return None

    # passage direction 계산
    cycle_points = points_full[verts]
    center = cycle_points.mean(axis=0)
    cycle_centered = cycle_points - center
    local_dim = min(len(verts) - 1, 10)
    local_pca = PCA(n_components=local_dim)
    local_pca.fit(cycle_centered)

    cycle_plane = local_pca.components_.T
    P_plane = cycle_plane @ cycle_plane.T
    P_orth = np.eye(points_full.shape[1]) - P_plane

    centered_all = points_full - center
    projected = centered_all @ P_orth.T
    cov = np.cov(projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    direction = None
    for i, ev in enumerate(eigenvalues):
        if ev > 1e-10:
            direction = eigenvectors[:, i]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            break

    return {
        'vertex_indices': verts,
        'persistence': best_pers,
        'center': center,
        'direction': direction,
        'birth': float(dgm[best_idx][0]),
        'death': float(dgm[best_idx][1]),
    }


def nonuniform_perturbation(
    embeddings: np.ndarray,
    wall_info: dict,
    alpha: float,
    mode: str = "cycle_only",
) -> np.ndarray:
    """
    비균일 섭동 전략.

    mode:
      "cycle_only"  — cycle 꼭짓점만 이동 (가장 정밀)
      "proximity"   — 벽 중심에 가까운 점일수록 더 많이 이동
      "radial"      — cycle 중심에서 바깥으로 밀기 (hole 닫기)
    """
    perturbed = embeddings.copy()
    direction = wall_info['direction']
    verts = wall_info['vertex_indices']
    center = wall_info['center']

    if mode == "cycle_only":
        # cycle 꼭짓점만 passage direction으로 이동
        for v in verts:
            perturbed[v] = perturbed[v] + alpha * direction

    elif mode == "proximity":
        # 벽 중심에 가까울수록 더 많이 이동 (가우시안 가중치)
        dists_to_center = np.linalg.norm(embeddings - center, axis=1)
        sigma = np.median(dists_to_center)
        weights = np.exp(-0.5 * (dists_to_center / sigma) ** 2)
        for i in range(len(embeddings)):
            perturbed[i] = perturbed[i] + alpha * weights[i] * direction

    elif mode == "radial":
        # cycle 꼭짓점을 중심에서 바깥으로 밀기 → hole 닫기
        for v in verts:
            radial = perturbed[v] - center
            radial_norm = np.linalg.norm(radial)
            if radial_norm > 1e-10:
                radial_unit = radial / radial_norm
                # 중심 쪽으로 당기기 (hole 수축)
                perturbed[v] = perturbed[v] - alpha * radial_unit

    return perturbed


def run():
    llm = load_model()

    prompts = [
        ("factual", "The capital of France is"),
        ("creative", "A color that doesn't exist yet would look like"),
        ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
    ]

    alpha_values = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
    modes = ["cycle_only", "proximity", "radial"]

    all_results = {}

    for category, prompt in prompts:
        print(f"\n{'='*70}")
        print(f"[{category}] \"{prompt}\"")
        print(f"{'='*70}")

        embeddings = extract_embeddings(llm, prompt)
        print(f"  Embeddings: {embeddings.shape}")

        result_orig, reduced_orig = full_topology(embeddings)
        orig_summary = topo_summary(result_orig)
        print(f"  Original: β₁={orig_summary['beta1']}, max_pers={orig_summary['max_pers']:.4f}")

        wall_info = extract_wall_info(result_orig, embeddings, reduced_orig)
        if wall_info is None:
            print("  [NO WALL] Skipping.")
            continue

        print(f"  Wall: pers={wall_info['persistence']:.4f}, "
              f"cycle_verts={len(wall_info['vertex_indices'])}, "
              f"birth={wall_info['birth']:.3f}, death={wall_info['death']:.3f}")

        cat_results = {}

        for mode in modes:
            print(f"\n  Mode: {mode}")
            print(f"  {'α':>6} {'β₁':>5} {'max_pers':>10} {'Δβ₁':>5} {'Δmax_pers':>11} {'Status':>12}")
            print(f"  {'-'*52}")

            mode_results = []

            for alpha in alpha_values:
                if alpha == 0.0:
                    summary = orig_summary
                else:
                    perturbed = nonuniform_perturbation(embeddings, wall_info, alpha, mode)
                    result_new, _ = full_topology(perturbed)
                    summary = topo_summary(result_new)

                delta_b1 = summary['beta1'] - orig_summary['beta1']
                delta_mp = summary['max_pers'] - orig_summary['max_pers']
                passed = summary['beta1'] < orig_summary['beta1']
                status = "★ PASSED" if passed else ("REDUCED" if delta_b1 < 0 else "-")

                print(f"  {alpha:>6.1f} {summary['beta1']:>5} {summary['max_pers']:>10.4f} "
                      f"{delta_b1:>5} {delta_mp:>11.4f} {status:>12}")

                mode_results.append({
                    'alpha': alpha,
                    'beta1': summary['beta1'],
                    'max_pers': summary['max_pers'],
                    'delta_beta1': delta_b1,
                    'delta_max_pers': delta_mp,
                    'wall_passed': passed,
                })

            cat_results[mode] = mode_results

        all_results[category] = cat_results

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(OUTPUT_DIR / "adapter_nonuniform_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    # 최종 요약
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY: WALL PASSAGE BY MODE")
    print(f"{'='*70}\n")

    for cat, modes_data in all_results.items():
        print(f"  [{cat}]")
        for mode, results in modes_data.items():
            passed = sum(1 for r in results if r['wall_passed'])
            total = sum(1 for r in results if r['alpha'] > 0)
            min_beta1 = min(r['beta1'] for r in results)
            print(f"    {mode:<15} passed: {passed}/{total}, min β₁ reached: {min_beta1}")
        print()


if __name__ == "__main__":
    run()
