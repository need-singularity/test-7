"""
PoC: 실제 Llama 3.1 8B (GGUF) hidden states에서 β₁ hole 감지

GGUF 양자화 모델에서 hidden states를 추출하고
persistent homology로 위상 구조를 분석.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

PCA_DIM = 50
MAX_POINTS = 150
PERSISTENCE_THRESHOLD = 0.01

PROMPTS = [
    # 사실 기반 (분포 내부)
    "The capital of France is",
    "Water boils at a temperature of",

    # 추론 (분포 경계)
    "If all roses are flowers and some flowers fade quickly, then",
    "The number that comes after the largest known prime is",

    # 창의적 (분포 바깥)
    "A color that doesn't exist yet would look like",
    "If mathematics were a living organism, its heartbeat would be",

    # 지식 한계
    "The solution to the Riemann hypothesis involves",
    "The mechanism by which consciousness emerges from neurons is",
]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"


def load_model():
    """GGUF 모델 로딩"""
    from llama_cpp import Llama

    print(f"Loading {MODEL_PATH.name}...")
    t0 = time.time()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_gpu_layers=-1,  # 가능하면 전부 GPU로
        embedding=True,
        verbose=False,
    )

    print(f"Model loaded in {time.time()-t0:.1f}s")
    return llm


def extract_embeddings(llm, prompt):
    """
    GGUF 모델에서 임베딩 추출.

    llama-cpp-python의 embed()는 마지막 레이어 임베딩을 반환.
    여러 토큰의 임베딩을 점 구름으로 사용.
    """
    # 토큰화
    tokens = llm.tokenize(prompt.encode('utf-8'))
    n_tokens = len(tokens)

    # 각 토큰 prefix까지의 임베딩 수집 (컨텍스트 변화에 따른 표현 변화)
    embeddings = []

    # 방법 1: 전체 시퀀스의 임베딩
    emb = llm.embed(prompt)
    if isinstance(emb[0], list):
        # 토큰별 임베딩
        embeddings = np.array(emb)
    else:
        # 단일 임베딩 — 점이 하나뿐이면 PH 불가
        embeddings = np.array([emb])

    return embeddings, n_tokens


def extract_embeddings_varied(llm, prompt, n_variations=30):
    """
    단일 프롬프트에서 다양한 점 구름 생성.

    전략:
    1. 프롬프트의 prefix들 (누적 토큰)
    2. 프롬프트 + 랜덤 suffix (분포 탐색)
    3. 약간 변형된 프롬프트들
    """
    points = []

    # 1. 원본 프롬프트 임베딩
    emb = llm.embed(prompt)
    if isinstance(emb[0], list):
        points.extend(emb)
    else:
        points.append(emb)

    # 2. Prefix 임베딩: 토큰을 하나씩 추가하며 임베딩 변화 추적
    tokens = llm.tokenize(prompt.encode('utf-8'))
    for i in range(1, len(tokens)):
        prefix = llm.detokenize(tokens[:i]).decode('utf-8', errors='replace')
        if prefix.strip():
            emb = llm.embed(prefix)
            if isinstance(emb[0], list):
                points.append(emb[-1])  # 마지막 토큰 임베딩
            else:
                points.append(emb)

    # 3. 프롬프트 + continuation 시도 (분포 경계 탐색)
    suffixes = [
        " and", " but", " however", " because", " which",
        " the", " a", " in", " of", " that",
        ".", "?", "!", " not", " never",
        " always", " perhaps", " certainly", " impossible", " undefined",
    ]
    for suffix in suffixes:
        extended = prompt + suffix
        emb = llm.embed(extended)
        if isinstance(emb[0], list):
            points.append(emb[-1])
        else:
            points.append(emb)

    points = np.array(points)
    print(f"    Collected {points.shape[0]} embedding points, dim={points.shape[1] if len(points.shape)>1 else 'scalar'}")
    return points


def subsample_fps(points, max_points=MAX_POINTS):
    """Farthest Point Sampling"""
    n = points.shape[0]
    if n <= max_points:
        return points

    indices = [0]
    min_dists = np.full(n, np.inf)
    for _ in range(max_points - 1):
        last = indices[-1]
        dists = np.linalg.norm(points - points[last], axis=1)
        min_dists = np.minimum(min_dists, dists)
        indices.append(np.argmax(min_dists))

    return points[indices]


def analyze_point_cloud(points, label=""):
    """점 구름 → PCA → PH → 위상 요약"""
    from ripser import ripser

    print(f"  [{label}] shape={points.shape}")

    # 서브샘플
    points = subsample_fps(points)

    # PCA
    n, d = points.shape
    pca_target = min(PCA_DIM, n - 1, d)
    if pca_target >= 2:
        pca = PCA(n_components=pca_target)
        reduced = pca.fit_transform(points)
        explained = sum(pca.explained_variance_ratio_)
        print(f"    PCA {d}→{pca_target}, explained={explained:.3f}")
    else:
        reduced = points

    # 거리 행렬
    dm = squareform(pdist(reduced))

    # PH
    t0 = time.time()
    result = ripser(dm, maxdim=2, distance_matrix=True)
    t_ph = time.time() - t0

    # 분석
    summary = {'ph_time': t_ph}
    for dim in range(3):
        dgm = result['dgms'][dim]
        bars = [(float(b), float(d), float(d-b)) for b, d in dgm
                if np.isfinite(d) and d - b > PERSISTENCE_THRESHOLD]
        bars.sort(key=lambda x: x[2], reverse=True)

        summary[f'beta_{dim}'] = len(bars)
        summary[f'beta_{dim}_bars'] = bars[:5]  # top 5
        summary[f'beta_{dim}_max_pers'] = bars[0][2] if bars else 0.0

    summary['has_wall'] = summary['beta_1'] > 0
    summary['wall_strength'] = summary['beta_1_max_pers']

    # 출력
    wall_str = f"★ WALL β₁={summary['beta_1']}, strength={summary['wall_strength']:.4f}" if summary['has_wall'] else "no wall"
    print(f"    β₀={summary['beta_0']}, β₁={summary['beta_1']}, β₂={summary['beta_2']} | {wall_str} | {t_ph:.2f}s")

    return summary


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    llm = load_model()

    all_results = {}

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(PROMPTS)}] \"{prompt}\"")
        print(f"{'='*60}")

        t0 = time.time()
        points = extract_embeddings_varied(llm, prompt)
        t_extract = time.time() - t0
        print(f"  Embedding extraction: {t_extract:.1f}s")

        if len(points.shape) < 2 or points.shape[0] < 3:
            print(f"  [SKIP] Not enough points: {points.shape}")
            continue

        summary = analyze_point_cloud(points, label=f"prompt_{i}")

        all_results[f'prompt_{i}'] = {
            'prompt': prompt,
            'n_points': int(points.shape[0]),
            'embedding_dim': int(points.shape[1]),
            **summary,
        }

    # 저장
    output_file = OUTPUT_DIR / "llama_topology_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    # 요약
    print(f"\n\n{'='*60}")
    print("LLAMA 8B HIDDEN STATE TOPOLOGY RESULTS")
    print(f"{'='*60}\n")
    print(f"{'Prompt':<55} {'β₁':>4} {'Wall?':>6} {'Strength':>10}")
    print("-" * 80)

    wall_count = 0
    for key, data in all_results.items():
        prompt_short = data['prompt'][:52] + "..." if len(data['prompt']) > 52 else data['prompt']
        has_wall = "YES" if data['has_wall'] else "no"
        strength = f"{data['wall_strength']:.4f}" if data['has_wall'] else "-"
        print(f"{prompt_short:<55} {data['beta_1']:>4} {has_wall:>6} {strength:>10}")
        if data['has_wall']:
            wall_count += 1

    total = len(all_results)
    print(f"\n벽 감지율: {wall_count}/{total} ({100*wall_count/total:.1f}%)")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run()
