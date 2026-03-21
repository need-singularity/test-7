"""
Phase 4: Emergence Score 기반 최적 벽 통과 탐색

GGUF에서는 gradient descent 불가 → 탐색 기반 최적화.

전략:
  1. 프롬프트별로 radial alpha를 탐색
  2. TECS EmergenceDetector로 각 alpha 평가
  3. 목표: β₁ 최소화 + emergence score 최대화 + 구조 안정성 유지
  4. 최적 alpha → Phase 5 생성 품질 비교에 사용
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"
PERSISTENCE_THRESHOLD = 0.01


# ── 공통 함수 ────────────────────────────────────────

def load_model():
    from llama_cpp import Llama
    print("Loading model...")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=512, n_gpu_layers=-1, embedding=True, verbose=False)
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


def compute_full_topology(points):
    from ripser import ripser
    n, d = points.shape
    pca_dim = min(50, n - 1, d)
    if pca_dim >= 2:
        reduced = PCA(n_components=pca_dim).fit_transform(points)
    else:
        reduced = points

    dm = squareform(pdist(reduced))
    result = ripser(dm, maxdim=1, distance_matrix=True, do_cocycles=True)

    dgm1 = result['dgms'][1]
    cocycles = result['cocycles'][1]

    walls = []
    for idx, (b, d_) in enumerate(dgm1):
        p = d_ - b
        if np.isfinite(d_) and p > PERSISTENCE_THRESHOLD:
            cc = cocycles[idx]
            verts = sorted(set(int(r[0]) for r in cc) | set(int(r[1]) for r in cc))
            walls.append({'birth': float(b), 'death': float(d_), 'persistence': float(p),
                          'vertex_indices': verts, 'idx': idx})
    walls.sort(key=lambda w: w['persistence'], reverse=True)

    beta0_bars = [(float(b), float(d_)) for b, d_ in result['dgms'][0]
                  if np.isfinite(d_) and d_ - b > PERSISTENCE_THRESHOLD]

    return {
        'beta0': len(beta0_bars) + sum(1 for b, d_ in result['dgms'][0] if not np.isfinite(d_)),
        'beta1': len(walls),
        'walls': walls,
        'max_persistence': walls[0]['persistence'] if walls else 0.0,
        'total_persistence': sum(w['persistence'] for w in walls),
        'mean_persistence': np.mean([w['persistence'] for w in walls]) if walls else 0.0,
        'result': result,
    }


def radial_perturbation(embeddings, wall, alpha):
    """단일 wall에 대한 radial 수축"""
    perturbed = embeddings.copy()
    verts = wall['vertex_indices']
    cycle_points = embeddings[verts]
    center = cycle_points.mean(axis=0)

    for v in verts:
        radial = perturbed[v] - center
        norm = np.linalg.norm(radial)
        if norm > 1e-10:
            perturbed[v] = perturbed[v] - alpha * (radial / norm)
    return perturbed


def multi_wall_perturbation(embeddings, walls, alpha, max_walls=None):
    """여러 wall을 동시에 수축 — 가장 persistent한 것부터"""
    perturbed = embeddings.copy()
    target_walls = walls[:max_walls] if max_walls else walls

    for wall in target_walls:
        verts = wall['vertex_indices']
        cycle_points = perturbed[verts]  # 이미 변형된 점 사용
        center = cycle_points.mean(axis=0)

        for v in verts:
            radial = perturbed[v] - center
            norm = np.linalg.norm(radial)
            if norm > 1e-10:
                perturbed[v] = perturbed[v] - alpha * (radial / norm)

    return perturbed


# ── Emergence Score ──────────────────────────────────

def emergence_score(topo_before, topo_after):
    """
    TECS EmergenceDetector 로직 + 커스텀 벽 통과 점수.

    3채널 점수:
      1. wall_reduction: β₁이 줄었나 (0~1)
      2. persistence_reduction: max persistence가 줄었나 (0~1)
      3. stability: β₀가 유지되었나 (1이면 안정, 0이면 붕괴)
    """
    from tecs.emergence import EmergenceDetector

    # TECS emergence
    detector = EmergenceDetector()
    topo_b = {'beta1': topo_before['beta1'], 'max_persistence_h1': topo_before['max_persistence']}
    topo_a = {'beta1': topo_after['beta1'], 'max_persistence_h1': topo_after['max_persistence']}
    hier = {'hierarchy_score': 0.5}

    score_before = detector.score(topo_b, hier)
    score_after = detector.score(topo_a, hier)

    # 벽 통과 전용 점수
    b1_before = topo_before['beta1']
    b1_after = topo_after['beta1']

    # wall reduction: β₁이 줄수록 좋음 (0→1)
    if b1_before > 0:
        wall_reduction = max(0.0, (b1_before - b1_after) / b1_before)
    else:
        wall_reduction = 1.0  # 이미 벽 없음

    # persistence reduction: max_pers가 줄수록 좋음
    mp_before = topo_before['max_persistence']
    mp_after = topo_after['max_persistence']
    if mp_before > 0:
        pers_reduction = max(0.0, (mp_before - mp_after) / mp_before)
    else:
        pers_reduction = 1.0

    # stability: β₀ 변화 없으면 1, 크게 바뀌면 0
    b0_before = topo_before['beta0']
    b0_after = topo_after['beta0']
    b0_change = abs(b0_after - b0_before)
    stability = max(0.0, 1.0 - b0_change / max(b0_before, 1))

    # 종합 점수 (가중합)
    passage_score = (
        0.4 * wall_reduction +
        0.3 * pers_reduction +
        0.3 * stability
    )

    return {
        'passage_score': round(passage_score, 4),
        'wall_reduction': round(wall_reduction, 4),
        'pers_reduction': round(pers_reduction, 4),
        'stability': round(stability, 4),
        'tecs_before': score_before.to_dict(),
        'tecs_after': score_after.to_dict(),
        'beta1_change': b1_after - b1_before,
        'max_pers_change': mp_after - mp_before,
    }


# ── 최적화 루프 ──────────────────────────────────────

@dataclass
class OptimizationResult:
    prompt: str
    category: str
    best_alpha: float
    best_score: float
    best_beta1: int
    original_beta1: int
    search_history: list


def optimize_alpha(
    embeddings: np.ndarray,
    topo_orig: dict,
    alpha_range: np.ndarray = None,
    max_walls: int = None,
) -> Tuple[float, float, list]:
    """
    최적 alpha 탐색 (grid search → refinement).

    Returns: (best_alpha, best_score, history)
    """
    if alpha_range is None:
        # 2단계 탐색: 거친 격자 → 세밀 격자
        alpha_range = np.concatenate([
            np.arange(0, 20, 2),
            np.arange(20, 55, 5),
        ])

    walls = topo_orig['walls']
    history = []

    best_alpha = 0.0
    best_score = 0.0

    for alpha in alpha_range:
        if alpha == 0:
            score_dict = emergence_score(topo_orig, topo_orig)
        else:
            perturbed = multi_wall_perturbation(embeddings, walls, alpha, max_walls)
            topo_new = compute_full_topology(perturbed)
            score_dict = emergence_score(topo_orig, topo_new)

        entry = {
            'alpha': float(alpha),
            **score_dict,
        }
        history.append(entry)

        if score_dict['passage_score'] > best_score:
            best_score = score_dict['passage_score']
            best_alpha = float(alpha)

    # 2단계: best 근처 세밀 탐색
    if best_alpha > 0:
        fine_range = np.linspace(max(0, best_alpha - 3), best_alpha + 3, 7)
        for alpha in fine_range:
            if alpha <= 0 or any(abs(h['alpha'] - alpha) < 0.1 for h in history):
                continue
            perturbed = multi_wall_perturbation(embeddings, walls, alpha, max_walls)
            topo_new = compute_full_topology(perturbed)
            score_dict = emergence_score(topo_orig, topo_new)

            entry = {'alpha': float(alpha), **score_dict}
            history.append(entry)

            if score_dict['passage_score'] > best_score:
                best_score = score_dict['passage_score']
                best_alpha = float(alpha)

    history.sort(key=lambda h: h['alpha'])
    return best_alpha, best_score, history


def run():
    llm = load_model()

    prompts = [
        ("factual", "The capital of France is"),
        ("factual2", "Water boils at a temperature of"),
        ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
        ("creative", "A color that doesn't exist yet would look like"),
        ("creative2", "If mathematics were a living organism, its heartbeat would be"),
        ("boundary", "The solution to the Riemann hypothesis involves"),
        ("boundary2", "The mechanism by which consciousness emerges from neurons is"),
    ]

    all_results = {}
    optimization_results = []

    for category, prompt in prompts:
        print(f"\n{'='*70}")
        print(f"[{category}] \"{prompt}\"")
        print(f"{'='*70}")

        t0 = time.time()
        embeddings = extract_embeddings(llm, prompt)
        print(f"  Embeddings: {embeddings.shape} ({time.time()-t0:.1f}s)")

        topo_orig = compute_full_topology(embeddings)
        print(f"  Original: β₁={topo_orig['beta1']}, max_pers={topo_orig['max_persistence']:.4f}")

        if topo_orig['beta1'] == 0:
            print("  [NO WALLS] Skipping optimization.")
            continue

        # 최적화: 전체 wall 동시 수축
        print(f"  Optimizing (multi-wall radial)...")
        t0 = time.time()
        best_alpha, best_score, history = optimize_alpha(embeddings, topo_orig)
        t_opt = time.time() - t0

        # 최적 alpha에서의 상세 결과
        perturbed = multi_wall_perturbation(embeddings, topo_orig['walls'], best_alpha)
        topo_best = compute_full_topology(perturbed)
        final_score = emergence_score(topo_orig, topo_best)

        print(f"  Optimization done in {t_opt:.1f}s")
        print(f"  ★ Best α={best_alpha:.1f}")
        print(f"    passage_score = {best_score:.4f}")
        print(f"    β₁: {topo_orig['beta1']} → {topo_best['beta1']}")
        print(f"    max_pers: {topo_orig['max_persistence']:.4f} → {topo_best['max_persistence']:.4f}")
        print(f"    wall_reduction = {final_score['wall_reduction']:.4f}")
        print(f"    pers_reduction = {final_score['pers_reduction']:.4f}")
        print(f"    stability = {final_score['stability']:.4f}")

        opt_result = OptimizationResult(
            prompt=prompt,
            category=category,
            best_alpha=best_alpha,
            best_score=best_score,
            best_beta1=topo_best['beta1'],
            original_beta1=topo_orig['beta1'],
            search_history=history,
        )
        optimization_results.append(opt_result)

        all_results[category] = {
            'prompt': prompt,
            'original_beta1': topo_orig['beta1'],
            'original_max_pers': topo_orig['max_persistence'],
            'best_alpha': best_alpha,
            'best_score': best_score,
            'best_beta1': topo_best['beta1'],
            'best_max_pers': topo_best['max_persistence'],
            'final_emergence': final_score,
            'history': history,
        }

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(OUTPUT_DIR / "phase4_optimization_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    # 요약
    print(f"\n\n{'='*70}")
    print("PHASE 4: EMERGENCE-GUIDED OPTIMIZATION RESULTS")
    print(f"{'='*70}\n")

    print(f"{'Category':<12} {'β₁ orig':>8} {'β₁ best':>8} {'Best α':>7} {'Score':>7} "
          f"{'WallRed':>8} {'PersRed':>8} {'Stable':>7}")
    print("-" * 75)

    for cat, data in all_results.items():
        e = data['final_emergence']
        print(f"{cat:<12} {data['original_beta1']:>8} {data['best_beta1']:>8} "
              f"{data['best_alpha']:>7.1f} {data['best_score']:>7.4f} "
              f"{e['wall_reduction']:>8.4f} {e['pers_reduction']:>8.4f} {e['stability']:>7.4f}")

    # α 탐색 곡선 출력 (각 카테고리별 최고 score 지점)
    print(f"\n\nα Search Curves (passage_score by α):")
    for cat, data in all_results.items():
        scores = [(h['alpha'], h['passage_score']) for h in data['history']]
        scores.sort()
        curve = " ".join(f"{a:.0f}:{s:.2f}" for a, s in scores[:10])
        print(f"  {cat}: {curve}")

    print(f"\nSaved to {OUTPUT_DIR / 'phase4_optimization_results.json'}")


if __name__ == "__main__":
    run()
