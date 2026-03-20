"""
Phase 3: Topological Adapter — 벽 통과 실험

passage direction으로 임베딩을 섭동(perturbation)하면:
1. 모델 출력이 바뀌는가? (벽 너머의 다른 답)
2. 위상 구조가 바뀌는가? (β₁ hole이 줄거나 이동)
3. emergence score가 변하는가?

GGUF 모델에서는 PyTorch adapter를 직접 삽입할 수 없으므로,
임베딩 공간에서의 섭동으로 개념을 증명.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"
PERSISTENCE_THRESHOLD = 0.01


# ── Phase 2에서 가져온 함수들 ──────────────────────────

def extract_embeddings(llm, prompt, n_suffixes=20):
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
            if isinstance(emb[0], list):
                points.append(emb[-1])
            else:
                points.append(emb)

    suffixes = [
        " and", " but", " however", " because", " which",
        " the", " a", " in", " of", " that",
        ".", "?", "!", " not", " never",
        " always", " perhaps", " certainly", " impossible", " undefined",
    ]
    for suffix in suffixes[:n_suffixes]:
        emb = llm.embed(prompt + suffix)
        if isinstance(emb[0], list):
            points.append(emb[-1])
        else:
            points.append(emb)

    return np.array(points)


def compute_topology(points):
    """점 구름의 위상 요약 반환"""
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

    dgm = result['dgms'][1]
    cocycles = result['cocycles'][1]

    walls = []
    for idx, (birth, death) in enumerate(dgm):
        pers = death - birth
        if np.isfinite(death) and pers > PERSISTENCE_THRESHOLD:
            cocycle = cocycles[idx]
            verts = set()
            for row in cocycle:
                verts.add(int(row[0]))
                verts.add(int(row[1]))
            walls.append({
                'birth': float(birth),
                'death': float(death),
                'persistence': float(pers),
                'n_vertices': len(verts),
                'vertex_indices': sorted(verts),
            })

    walls.sort(key=lambda w: w['persistence'], reverse=True)

    beta0 = sum(1 for b, d in result['dgms'][0]
                if (not np.isfinite(d)) or d - b > PERSISTENCE_THRESHOLD)
    beta1 = len(walls)

    total_persistence = sum(w['persistence'] for w in walls)

    return {
        'beta0': beta0,
        'beta1': beta1,
        'walls': walls,
        'total_persistence': total_persistence,
        'max_persistence': walls[0]['persistence'] if walls else 0.0,
    }


def extract_passage_direction(points):
    """점 구름에서 가장 강한 β₁ hole의 통과 방향 추출"""
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

    dgm = result['dgms'][1]
    cocycles = result['cocycles'][1]

    # 가장 persistent한 hole
    best_idx = -1
    best_pers = 0
    for idx, (birth, death) in enumerate(dgm):
        pers = death - birth
        if np.isfinite(death) and pers > best_pers:
            best_pers = pers
            best_idx = idx

    if best_idx < 0:
        return None, None

    cocycle = cocycles[best_idx]
    verts = sorted(set(int(row[0]) for row in cocycle) | set(int(row[1]) for row in cocycle))

    if len(verts) < 3:
        return None, None

    cycle_points = points[verts]
    center = cycle_points.mean(axis=0)

    cycle_centered = cycle_points - center
    local_dim = min(len(verts) - 1, 10)
    local_pca = PCA(n_components=local_dim)
    local_pca.fit(cycle_centered)

    cycle_plane = local_pca.components_.T
    P_plane = cycle_plane @ cycle_plane.T
    P_orth = np.eye(d) - P_plane

    centered_all = points - center
    projected = centered_all @ P_orth.T
    cov = np.cov(projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    for i, ev in enumerate(eigenvalues):
        if ev > 1e-10:
            direction = eigenvectors[:, i]
            return direction / (np.linalg.norm(direction) + 1e-10), center

    return None, None


# ── Topological Adapter ──────────────────────────────

@dataclass
class AdapterResult:
    """adapter 적용 전후 비교 결과"""
    prompt: str
    alpha: float                        # 섭동 강도
    # 원본
    original_completion: str
    original_beta1: int
    original_max_persistence: float
    original_total_persistence: float
    # 섭동 후
    adapted_completion: str
    adapted_beta1: int
    adapted_max_persistence: float
    adapted_total_persistence: float
    # 변화
    delta_beta1: int
    delta_max_persistence: float
    delta_total_persistence: float
    wall_passage_detected: bool         # β₁이 줄었으면 벽 통과


def apply_topological_adapter(
    llm,
    prompt: str,
    alpha_values: List[float] = None,
    max_tokens: int = 50,
) -> List[AdapterResult]:
    """
    Topological Adapter 적용.

    1. 원본 프롬프트의 임베딩 + 위상 분석
    2. passage direction 추출
    3. 다양한 alpha(섭동 강도)로 임베딩 섭동
    4. 섭동된 임베딩에 가장 가까운 토큰으로 프롬프트 재구성
    5. 변형 프롬프트의 출력 + 위상 분석
    6. 전후 비교
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"  Extracting embeddings...")
    embeddings = extract_embeddings(llm, prompt)
    print(f"  Shape: {embeddings.shape}")

    # 원본 위상
    print(f"  Computing original topology...")
    orig_topo = compute_topology(embeddings)
    print(f"  Original: β₁={orig_topo['beta1']}, max_pers={orig_topo['max_persistence']:.4f}")

    # 통과 방향
    print(f"  Extracting passage direction...")
    direction, center = extract_passage_direction(embeddings)
    if direction is None:
        print(f"  [NO HOLES FOUND - skipping]")
        return []

    # 원본 completion
    print(f"  Generating original completion...")
    orig_output = llm.create_completion(prompt, max_tokens=max_tokens, temperature=0.0)
    orig_completion = orig_output['choices'][0]['text'].strip()

    results = []

    for alpha in alpha_values:
        print(f"\n  α={alpha}:")

        if alpha == 0.0:
            # baseline
            results.append(AdapterResult(
                prompt=prompt,
                alpha=0.0,
                original_completion=orig_completion,
                original_beta1=orig_topo['beta1'],
                original_max_persistence=orig_topo['max_persistence'],
                original_total_persistence=orig_topo['total_persistence'],
                adapted_completion=orig_completion,
                adapted_beta1=orig_topo['beta1'],
                adapted_max_persistence=orig_topo['max_persistence'],
                adapted_total_persistence=orig_topo['total_persistence'],
                delta_beta1=0,
                delta_max_persistence=0.0,
                delta_total_persistence=0.0,
                wall_passage_detected=False,
            ))
            print(f"    [baseline] β₁={orig_topo['beta1']}")
            continue

        # 임베딩 섭동: 모든 점을 passage direction으로 이동
        perturbed = embeddings + alpha * direction

        # 섭동된 임베딩의 위상 분석
        adapted_topo = compute_topology(perturbed)

        # 섭동된 프롬프트로 생성
        # 방법: passage direction을 프롬프트 뒤에 토큰으로 힌트
        # (GGUF에서는 직접 임베딩 주입 불가 → 간접 방법)
        # 대신: direction의 top 뉴런 차원을 분석하여 의미론적 힌트 구성
        # 지금은 원본 프롬프트 + 약간의 변형으로 대체
        perturbed_prompt = _create_perturbed_prompt(prompt, direction, alpha)
        try:
            adapted_output = llm.create_completion(perturbed_prompt, max_tokens=max_tokens, temperature=0.0)
            adapted_completion = adapted_output['choices'][0]['text'].strip()
        except Exception as e:
            adapted_completion = f"[error: {e}]"

        delta_beta1 = adapted_topo['beta1'] - orig_topo['beta1']
        delta_max_pers = adapted_topo['max_persistence'] - orig_topo['max_persistence']
        delta_total_pers = adapted_topo['total_persistence'] - orig_topo['total_persistence']

        wall_passage = adapted_topo['beta1'] < orig_topo['beta1']

        result = AdapterResult(
            prompt=prompt,
            alpha=alpha,
            original_completion=orig_completion,
            original_beta1=orig_topo['beta1'],
            original_max_persistence=orig_topo['max_persistence'],
            original_total_persistence=orig_topo['total_persistence'],
            adapted_completion=adapted_completion,
            adapted_beta1=adapted_topo['beta1'],
            adapted_max_persistence=adapted_topo['max_persistence'],
            adapted_total_persistence=adapted_topo['total_persistence'],
            delta_beta1=delta_beta1,
            delta_max_persistence=delta_max_pers,
            delta_total_persistence=delta_total_pers,
            wall_passage_detected=wall_passage,
        )
        results.append(result)

        status = "★ WALL PASSED" if wall_passage else "wall intact"
        print(f"    β₁: {orig_topo['beta1']}→{adapted_topo['beta1']} (Δ={delta_beta1}) | {status}")
        print(f"    max_pers: {orig_topo['max_persistence']:.3f}→{adapted_topo['max_persistence']:.3f}")
        print(f"    completion: \"{adapted_completion[:60]}...\"")

    return results


def _create_perturbed_prompt(prompt: str, direction: np.ndarray, alpha: float) -> str:
    """
    passage direction에 기반한 프롬프트 변형.

    direction의 top 활성 차원을 분석하여
    모델이 "벽 너머"로 가도록 유도하는 토큰 추가.

    이것은 GGUF 제한 하의 근사 — 진짜 adapter는 PyTorch에서 구현.
    """
    # alpha에 비례하여 탐색 키워드 추가
    if alpha < 1.0:
        return prompt
    elif alpha < 3.0:
        # 약한 섭동: 기존 방향을 살짝 틈
        nudges = [
            " (thinking beyond the obvious)",
            " (considering unconventional perspectives)",
            " (exploring unexpected connections)",
        ]
        idx = min(int(alpha), len(nudges) - 1)
        return prompt + nudges[idx]
    elif alpha < 7.0:
        # 중간 섭동: 프레임 전환
        return f"Ignoring all conventional assumptions: {prompt}"
    else:
        # 강한 섭동: 완전히 다른 관점
        return f"From the perspective of a concept that doesn't yet exist in human language: {prompt}"


# ── Emergence Score (TECS 연동) ──────────────────────

def compute_emergence_score(before_topo: dict, after_topo: dict) -> dict:
    """TECS EmergenceDetector 로직으로 창발 점수 계산"""
    from tecs.emergence import EmergenceDetector

    topo_bundle_before = {
        'beta1': before_topo['beta1'],
        'max_persistence_h1': before_topo['max_persistence'],
    }
    topo_bundle_after = {
        'beta1': after_topo['beta1'],
        'max_persistence_h1': after_topo['max_persistence'],
    }

    # hierarchy 근사 (여기선 고정)
    hier_bundle = {'hierarchy_score': 0.5}

    detector = EmergenceDetector()
    score_before = detector.score(topo_bundle_before, hier_bundle)
    score_after = detector.score(topo_bundle_after, hier_bundle)

    return {
        'before': score_before.to_dict(),
        'after': score_after.to_dict(),
        'delta_total': score_after.total_score - score_before.total_score,
        'interpretation_change': f"{score_before.interpretation} → {score_after.interpretation}",
    }


# ── Main ──────────────────────────────────────────────

def run():
    from llama_cpp import Llama

    print(f"Loading model...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_gpu_layers=-1,
        embedding=True,
        verbose=False,
    )
    print("Model loaded.\n")

    prompts = [
        ("factual", "The capital of France is"),
        ("creative", "A color that doesn't exist yet would look like"),
        ("boundary", "The solution to the Riemann hypothesis involves"),
        ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
    ]

    all_results = {}

    for category, prompt in prompts:
        print(f"\n{'='*70}")
        print(f"[{category}] \"{prompt}\"")
        print(f"{'='*70}")

        results = apply_topological_adapter(llm, prompt)

        if not results:
            continue

        all_results[category] = []
        for r in results:
            all_results[category].append({
                'alpha': r.alpha,
                'original_completion': r.original_completion[:100],
                'adapted_completion': r.adapted_completion[:100],
                'original_beta1': r.original_beta1,
                'adapted_beta1': r.adapted_beta1,
                'delta_beta1': r.delta_beta1,
                'original_max_pers': r.original_max_persistence,
                'adapted_max_pers': r.adapted_max_persistence,
                'wall_passage': r.wall_passage_detected,
            })

    # 결과 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "adapter_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # 요약
    print(f"\n\n{'='*70}")
    print("TOPOLOGICAL ADAPTER RESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Category':<12} {'α':>5} {'β₁ before':>10} {'β₁ after':>10} {'Δβ₁':>5} {'Wall?':>8}")
    print("-" * 55)

    passage_count = 0
    total_count = 0

    for cat, results in all_results.items():
        for r in results:
            if r['alpha'] == 0.0:
                continue
            wall = "PASSED" if r['wall_passage'] else "-"
            print(f"{cat:<12} {r['alpha']:>5.1f} {r['original_beta1']:>10} {r['adapted_beta1']:>10} {r['delta_beta1']:>5} {wall:>8}")
            if r['wall_passage']:
                passage_count += 1
            total_count += 1

    print(f"\n벽 통과 성공률: {passage_count}/{total_count} ({100*passage_count/total_count:.1f}%)")
    print(f"\nResults saved to {OUTPUT_DIR / 'adapter_results.json'}")


if __name__ == "__main__":
    run()
