"""
PoC Phase 1: Llama 8B hidden states의 위상 구조 분석

목표: β₁ hole이 실제로 존재하는지 확인
- 다양한 프롬프트에 대해 hidden states 추출
- 점 구름(point cloud)으로 변환
- 차원 축소 후 persistent homology 계산
- β₁ > 0 인 경우 → "벽"이 존재함을 확인

GPU 메모리 ≥ 16GB 권장 (8B 모델 로딩)
CPU 모드도 지원 (느리지만 동작)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────

MODEL_ID = "meta-llama/Llama-3.1-8B"  # 또는 로컬 경로
MAX_NEW_TOKENS = 1
SAMPLE_LAYERS = [0, 8, 16, 24, 31]  # 분석할 레이어 인덱스
PCA_DIM = 50          # PH 계산 전 차원 축소 목표
MAX_POINTS = 200      # PH에 넣을 최대 점 수 (O(n⁴) 주의)
PERSISTENCE_THRESHOLD = 0.01

# 다양한 도메인의 프롬프트 — hidden states 패턴이 달라야 함
PROMPTS = [
    # 사실 기반 (분포 안쪽)
    "The capital of France is",
    "Water boils at a temperature of",
    "The speed of light in vacuum is approximately",

    # 추론 요구 (분포 경계)
    "If all roses are flowers and some flowers fade quickly, then",
    "The number that comes after the largest known prime is",

    # 창의적 / 분포 바깥
    "A color that doesn't exist yet would look like",
    "The opposite of gravity in a world where time flows backwards is",
    "If mathematics were a living organism, its heartbeat would be",

    # 지식 경계 (학습 데이터에 없을 가능성)
    "The solution to the Riemann hypothesis involves",
    "The mechanism by which consciousness emerges from neurons is",
]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"


def load_model():
    """Llama 8B 로딩 (float16, auto device)"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GPU 사용 가능하면 float16, 아니면 float32
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        output_hidden_states=True,
    )
    if device == "mps":
        model = model.to(device)

    model.eval()
    print(f"Model loaded on {device}, dtype={dtype}")
    return model, tokenizer, device


def extract_hidden_states(model, tokenizer, prompt, device):
    """프롬프트에 대한 모든 레이어의 hidden states 추출"""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states

    result = {}
    for layer_idx in SAMPLE_LAYERS:
        if layer_idx < len(hidden_states):
            hs = hidden_states[layer_idx][0].cpu().float().numpy()  # (seq_len, hidden_dim)
            result[layer_idx] = hs

    return result


def reduce_dimensions(points, target_dim=PCA_DIM):
    """PCA로 차원 축소 (4096 → 50)"""
    from sklearn.decomposition import PCA

    n_points, n_dims = points.shape
    actual_dim = min(target_dim, n_points - 1, n_dims)

    if actual_dim < 2:
        return points[:, :2] if n_dims >= 2 else points

    pca = PCA(n_components=actual_dim)
    reduced = pca.fit_transform(points)
    explained = sum(pca.explained_variance_ratio_)
    print(f"    PCA {n_dims}→{actual_dim}, explained variance: {explained:.3f}")
    return reduced


def compute_distance_matrix(points):
    """유클리드 거리 행렬 계산"""
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(points, metric='euclidean'))


def subsample_points(points, max_points=MAX_POINTS):
    """점이 너무 많으면 FPS(Farthest Point Sampling)로 서브샘플"""
    n = points.shape[0]
    if n <= max_points:
        return points

    # Farthest Point Sampling
    indices = [0]
    min_dists = np.full(n, np.inf)

    for _ in range(max_points - 1):
        last = indices[-1]
        dists = np.linalg.norm(points - points[last], axis=1)
        min_dists = np.minimum(min_dists, dists)
        next_idx = np.argmax(min_dists)
        indices.append(next_idx)

    print(f"    Subsampled {n} → {max_points} points (FPS)")
    return points[indices]


def compute_persistent_homology(dm):
    """
    거리 행렬로부터 persistent homology 계산.

    우선 순수 Python으로 구현 (TECS Rust 엔진 연동은 Phase 2).
    ripser 패키지가 있으면 사용, 없으면 scipy 기반 근사.
    """
    try:
        from ripser import ripser
        result = ripser(dm, maxdim=2, distance_matrix=True)
        diagrams = result['dgms']

        pairs = {0: [], 1: [], 2: []}
        for dim in range(min(3, len(diagrams))):
            for birth, death in diagrams[dim]:
                if np.isfinite(death):
                    pairs[dim].append({
                        'birth': float(birth),
                        'death': float(death),
                        'persistence': float(death - birth),
                    })

        return pairs, diagrams

    except ImportError:
        print("    [ripser not found — using TECS Rust engine or fallback]")
        return _fallback_ph(dm)


def _fallback_ph(dm):
    """ripser 없을 때의 간단한 VR complex 기반 근사"""
    n = dm.shape[0]

    # β₀ 근사: single-linkage clustering으로 connected components 추적
    from scipy.cluster.hierarchy import linkage, fcluster

    condensed = dm[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method='single')

    pairs = {0: [], 1: [], 2: []}

    # β₀ pairs: 클러스터가 합쳐지는 시점
    for i, (c1, c2, dist, _) in enumerate(Z):
        pairs[0].append({
            'birth': 0.0,
            'death': float(dist),
            'persistence': float(dist),
        })

    # β₁ 근사: 불가 (정확한 계산에는 boundary matrix 필요)
    # → ripser 설치 또는 TECS Rust 엔진 연동 필요
    print("    [WARNING] β₁/β₂ 계산에는 ripser 또는 TECS Rust 엔진이 필요합니다")

    return pairs, None


def analyze_topology(pairs):
    """위상 분석 결과 요약"""
    summary = {}

    for dim in [0, 1, 2]:
        dim_pairs = pairs.get(dim, [])
        significant = [p for p in dim_pairs if p['persistence'] > PERSISTENCE_THRESHOLD]

        summary[f'beta_{dim}_total'] = len(dim_pairs)
        summary[f'beta_{dim}_significant'] = len(significant)

        if significant:
            persistences = [p['persistence'] for p in significant]
            summary[f'beta_{dim}_max_persistence'] = max(persistences)
            summary[f'beta_{dim}_mean_persistence'] = np.mean(persistences)
            summary[f'beta_{dim}_bars'] = significant
        else:
            summary[f'beta_{dim}_max_persistence'] = 0.0
            summary[f'beta_{dim}_mean_persistence'] = 0.0
            summary[f'beta_{dim}_bars'] = []

    # 핵심 판단: β₁ hole이 존재하는가?
    summary['has_wall'] = summary['beta_1_significant'] > 0
    summary['wall_strength'] = summary['beta_1_max_persistence']

    return summary


def run_poc():
    """전체 PoC 파이프라인 실행"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model()

    all_results = {}

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"[{prompt_idx+1}/{len(PROMPTS)}] \"{prompt[:50]}...\"")
        print(f"{'='*60}")

        # 1. Hidden states 추출
        t0 = time.time()
        layer_states = extract_hidden_states(model, tokenizer, prompt, device)
        t_extract = time.time() - t0
        print(f"  Hidden states extracted in {t_extract:.2f}s")

        prompt_results = {}

        for layer_idx, states in layer_states.items():
            print(f"\n  Layer {layer_idx}: shape={states.shape}")

            # 2. 서브샘플 + 차원 축소
            points = subsample_points(states)
            reduced = reduce_dimensions(points)

            # 3. 거리 행렬
            dm = compute_distance_matrix(reduced)
            print(f"    Distance matrix: {dm.shape}, range=[{dm.min():.3f}, {dm.max():.3f}]")

            # 4. Persistent homology
            t0 = time.time()
            pairs, diagrams = compute_persistent_homology(dm)
            t_ph = time.time() - t0
            print(f"    PH computed in {t_ph:.2f}s")

            # 5. 분석
            summary = analyze_topology(pairs)
            print(f"    β₀={summary['beta_0_significant']}, "
                  f"β₁={summary['beta_1_significant']}, "
                  f"β₂={summary['beta_2_significant']}")

            if summary['has_wall']:
                print(f"    ★ WALL DETECTED! β₁ holes: {summary['beta_1_significant']}, "
                      f"max persistence: {summary['wall_strength']:.4f}")
            else:
                print(f"    (no significant β₁ holes)")

            prompt_results[f'layer_{layer_idx}'] = {
                'n_points': states.shape[0],
                'n_dims': states.shape[1],
                'reduced_dims': reduced.shape[1],
                **summary,
            }

        all_results[f'prompt_{prompt_idx}'] = {
            'prompt': prompt,
            'layers': prompt_results,
        }

    # 결과 저장
    output_file = OUTPUT_DIR / "poc_results.json"

    # numpy 타입 변환
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # 요약 출력
    print("\n## SUMMARY ##\n")
    print(f"{'Prompt':<55} {'Layer':>5} {'β₁':>4} {'Wall?':>6} {'Strength':>10}")
    print("-" * 85)

    wall_count = 0
    total_checks = 0

    for key, data in all_results.items():
        prompt_short = data['prompt'][:52] + "..." if len(data['prompt']) > 52 else data['prompt']
        for lkey, ldata in data['layers'].items():
            layer_num = lkey.replace('layer_', '')
            has_wall = "YES" if ldata['has_wall'] else "no"
            strength = f"{ldata['wall_strength']:.4f}" if ldata['has_wall'] else "-"
            beta1 = ldata['beta_1_significant']

            if ldata['has_wall']:
                wall_count += 1
            total_checks += 1

            print(f"{prompt_short:<55} {layer_num:>5} {beta1:>4} {has_wall:>6} {strength:>10}")
            prompt_short = ""  # 같은 프롬프트는 빈칸

    print(f"\n벽 감지율: {wall_count}/{total_checks} ({100*wall_count/total_checks:.1f}%)")

    return all_results


if __name__ == "__main__":
    run_poc()
