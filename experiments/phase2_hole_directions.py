"""
Phase 2: β₁ hole의 방향 벡터 추출

β₁ hole이 존재한다는 것은 확인됨 (Phase 1: 8/8).
이제 그 hole의 "방향"을 4096차원 벡터로 변환.

핵심 아이디어:
  1. ripser의 cocycle로 hole을 구성하는 edge들 식별
  2. 해당 edge의 꼭짓점(임베딩 벡터)들 수집
  3. cycle 점들의 PCA → cycle이 놓인 평면 식별
  4. 그 평면의 법선벡터 = "벽을 통과하는 방향"

벽을 통과한다 = 법선 방향으로 이동한다
(3차원에서 2차원 벽을 넘는 것과 동일한 원리)
"""

import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PERSISTENCE_THRESHOLD = 0.01


@dataclass
class WallInfo:
    """감지된 벽(β₁ hole)의 정보"""
    birth: float
    death: float
    persistence: float
    cycle_vertex_indices: List[int]       # cycle을 구성하는 점 인덱스
    cycle_plane_vectors: np.ndarray       # cycle이 놓인 평면의 기저 (k, embed_dim)
    passage_direction: np.ndarray          # 벽 통과 방향 (embed_dim,) — 법선벡터
    passage_direction_pca: np.ndarray      # PCA 공간에서의 통과 방향
    wall_center: np.ndarray                # cycle 중심점 (embed_dim,)
    explained_variance: float              # cycle 평면이 설명하는 분산 비율


@dataclass
class TopologyAnalysis:
    """프롬프트 하나에 대한 전체 위상 분석 결과"""
    prompt: str
    n_points: int
    embedding_dim: int
    walls: List[WallInfo]
    embeddings: np.ndarray                 # 원본 임베딩 (n_points, embed_dim)
    pca_embeddings: np.ndarray             # PCA 축소된 임베딩


def load_model():
    from llama_cpp import Llama
    print(f"Loading {MODEL_PATH.name}...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_gpu_layers=-1,
        embedding=True,
        verbose=False,
    )
    print("Model loaded.")
    return llm


def extract_embeddings(llm, prompt, n_suffixes=20):
    """프롬프트에서 다양한 임베딩 점 수집"""
    points = []

    # 원본
    emb = llm.embed(prompt)
    if isinstance(emb[0], list):
        points.extend(emb)
    else:
        points.append(emb)

    # Prefix 임베딩
    tokens = llm.tokenize(prompt.encode('utf-8'))
    for i in range(1, len(tokens)):
        prefix = llm.detokenize(tokens[:i]).decode('utf-8', errors='replace')
        if prefix.strip():
            emb = llm.embed(prefix)
            if isinstance(emb[0], list):
                points.append(emb[-1])
            else:
                points.append(emb)

    # Suffix 변형
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


def extract_hole_directions(
    embeddings: np.ndarray,
    pca_dim: int = 50,
    top_k_holes: int = 5,
) -> Tuple[List[WallInfo], np.ndarray]:
    """
    임베딩에서 β₁ hole의 방향 벡터를 추출.

    Returns:
        walls: WallInfo 리스트 (persistence 순 정렬)
        pca_embeddings: PCA 축소된 임베딩
    """
    from ripser import ripser

    n, d = embeddings.shape

    # PCA
    pca_target = min(pca_dim, n - 1, d)
    pca = PCA(n_components=pca_target)
    reduced = pca.fit_transform(embeddings)
    # PCA 역변환에 필요한 components 저장
    pca_components = pca.components_  # (pca_target, embed_dim)

    # 거리 행렬
    dm = squareform(pdist(reduced))

    # PH with cocycles
    result = ripser(dm, maxdim=1, distance_matrix=True, do_cocycles=True)

    dgm = result['dgms'][1]  # β₁ diagram
    cocycles = result['cocycles'][1]  # β₁ cocycles

    # significant holes 필터링
    walls = []
    for idx, (birth, death) in enumerate(dgm):
        persistence = death - birth
        if not np.isfinite(death) or persistence < PERSISTENCE_THRESHOLD:
            continue

        # cocycle에서 cycle 꼭짓점 추출
        cocycle = cocycles[idx]
        # cocycle: array of [edge_idx_i, edge_idx_j, coefficient]
        vertex_set = set()
        for row in cocycle:
            vertex_set.add(int(row[0]))
            vertex_set.add(int(row[1]))

        cycle_indices = sorted(vertex_set)

        if len(cycle_indices) < 3:
            continue

        # cycle 점들의 원본 임베딩
        cycle_points_full = embeddings[cycle_indices]  # (k, embed_dim)
        cycle_points_pca = reduced[cycle_indices]       # (k, pca_dim)

        # cycle 중심
        wall_center = cycle_points_full.mean(axis=0)  # (embed_dim,)

        # cycle 점들에 대해 로컬 PCA → cycle이 놓인 평면 찾기
        cycle_centered = cycle_points_full - wall_center
        k = len(cycle_indices)
        local_pca_dim = min(k - 1, 10)  # cycle 평면 차원

        if local_pca_dim < 1:
            continue

        local_pca = PCA(n_components=local_pca_dim)
        local_pca.fit(cycle_centered)

        # cycle 평면의 기저벡터들
        cycle_plane = local_pca.components_  # (local_pca_dim, embed_dim)
        explained = sum(local_pca.explained_variance_ratio_)

        # ★ 핵심: 통과 방향 = cycle 평면에 수직인 방향
        # 방법: 전체 임베딩 공간에서 cycle 평면에 직교하는 방향 찾기
        passage_dir = _compute_passage_direction(
            embeddings, wall_center, cycle_plane
        )

        # PCA 공간에서의 통과 방향
        passage_dir_pca = pca.transform(passage_dir.reshape(1, -1))[0] - pca.transform(wall_center.reshape(1, -1))[0]
        passage_dir_pca = passage_dir_pca / (np.linalg.norm(passage_dir_pca) + 1e-10)

        walls.append(WallInfo(
            birth=float(birth),
            death=float(death),
            persistence=float(persistence),
            cycle_vertex_indices=cycle_indices,
            cycle_plane_vectors=cycle_plane,
            passage_direction=passage_dir,
            passage_direction_pca=passage_dir_pca,
            wall_center=wall_center,
            explained_variance=float(explained),
        ))

    # persistence 순 정렬
    walls.sort(key=lambda w: w.persistence, reverse=True)
    walls = walls[:top_k_holes]

    return walls, reduced


def _compute_passage_direction(
    all_embeddings: np.ndarray,
    wall_center: np.ndarray,
    cycle_plane: np.ndarray,
) -> np.ndarray:
    """
    벽 통과 방향 계산.

    cycle 평면에 직교하면서, 전체 임베딩 분포에서 가장 "비어있는" 방향.

    방법:
    1. cycle 평면의 직교 보완 공간 계산
    2. 그 공간에서 데이터 밀도가 가장 낮은 방향 선택
       (= 분포가 가장 안 탐색한 방향 = "벽 너머")
    """
    embed_dim = all_embeddings.shape[1]

    # cycle 평면에 직교하는 방향들 구하기
    # Gram-Schmidt로 cycle_plane을 직교화 (이미 PCA라 거의 직교)
    Q = cycle_plane.T  # (embed_dim, local_pca_dim)

    # 투영 행렬: P = Q @ Q^T
    P_plane = Q @ Q.T  # (embed_dim, embed_dim)

    # 직교 보완 투영: P_orth = I - P_plane
    P_orth = np.eye(embed_dim) - P_plane

    # 전체 임베딩을 직교 보완 공간에 투영
    centered = all_embeddings - wall_center
    projected = centered @ P_orth.T  # (n, embed_dim)

    # 투영된 공간에서 분산이 가장 작은 방향 = 가장 비어있는 방향
    # (분산이 큰 방향은 이미 탐색된 방향)
    cov = np.cov(projected.T)

    # 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 가장 작은 비영(非零) 고유값의 고유벡터 = 가장 비어있는 방향
    # (영 고유값은 cycle 평면 자체)
    min_nonzero_idx = None
    for i, ev in enumerate(eigenvalues):
        if ev > 1e-10:
            min_nonzero_idx = i
            break

    if min_nonzero_idx is None:
        # fallback: 랜덤 직교 방향
        direction = np.random.randn(embed_dim)
        direction = direction - Q @ (Q.T @ direction)
        return direction / (np.linalg.norm(direction) + 1e-10)

    passage_dir = eigenvectors[:, min_nonzero_idx]
    passage_dir = passage_dir / (np.linalg.norm(passage_dir) + 1e-10)

    return passage_dir


def verify_passage_direction(wall: WallInfo, embeddings: np.ndarray):
    """
    통과 방향이 실제로 벽에 수직인지 검증.

    검증 기준:
    1. passage_direction과 cycle_plane_vectors의 내적 ≈ 0 (직교)
    2. passage_direction 방향으로의 데이터 분산이 작음 (비어있음)
    """
    # 직교성 검증
    dots = wall.cycle_plane_vectors @ wall.passage_direction
    max_dot = np.max(np.abs(dots))

    # 비어있음 검증
    centered = embeddings - wall.wall_center
    projections = centered @ wall.passage_direction
    variance_passage = np.var(projections)

    # cycle 평면 방향 분산과 비교
    plane_variances = []
    for vec in wall.cycle_plane_vectors:
        proj = centered @ vec
        plane_variances.append(np.var(proj))
    mean_plane_var = np.mean(plane_variances) if plane_variances else 0

    return {
        'orthogonality': float(max_dot),       # 0에 가까울수록 좋음
        'passage_variance': float(variance_passage),
        'plane_variance': float(mean_plane_var),
        'emptiness_ratio': float(variance_passage / (mean_plane_var + 1e-10)),  # 1보다 작을수록 비어있음
    }


def run():
    llm = load_model()

    prompts = [
        ("factual", "The capital of France is"),
        ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
        ("creative", "A color that doesn't exist yet would look like"),
        ("boundary", "The solution to the Riemann hypothesis involves"),
    ]

    print("\n" + "="*70)
    print("PHASE 2: HOLE DIRECTION EXTRACTION")
    print("="*70)

    for category, prompt in prompts:
        print(f"\n{'─'*70}")
        print(f"[{category}] \"{prompt}\"")
        print(f"{'─'*70}")

        # 임베딩 추출
        t0 = time.time()
        embeddings = extract_embeddings(llm, prompt)
        print(f"  Embeddings: {embeddings.shape} ({time.time()-t0:.1f}s)")

        # hole 방향 추출
        t0 = time.time()
        walls, pca_emb = extract_hole_directions(embeddings)
        print(f"  Walls found: {len(walls)} ({time.time()-t0:.1f}s)")

        for i, wall in enumerate(walls):
            print(f"\n  Wall #{i+1}:")
            print(f"    persistence = {wall.persistence:.4f} (birth={wall.birth:.3f}, death={wall.death:.3f})")
            print(f"    cycle vertices = {len(wall.cycle_vertex_indices)} points")
            print(f"    cycle plane explained = {wall.explained_variance:.3f}")
            print(f"    passage direction norm = {np.linalg.norm(wall.passage_direction):.4f}")

            # 검증
            v = verify_passage_direction(wall, embeddings)
            print(f"    ✓ orthogonality = {v['orthogonality']:.6f} (→0 good)")
            print(f"    ✓ emptiness ratio = {v['emptiness_ratio']:.4f} (<1 = empty)")

            # 통과 방향의 의미론적 해석
            # passage_direction의 가장 큰 성분 = 4096차원 중 어느 뉴런이 관여하는지
            top_dims = np.argsort(np.abs(wall.passage_direction))[-5:][::-1]
            top_vals = wall.passage_direction[top_dims]
            print(f"    top dims: {list(zip(top_dims.tolist(), [f'{v:.4f}' for v in top_vals]))}")

    # 통과 방향 저장
    output_dir = Path(__file__).parent.parent / "data" / "poc_topology"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 마지막 프롬프트의 결과를 npy로 저장 (adapter 학습용)
    if walls:
        directions = np.stack([w.passage_direction for w in walls])
        np.save(output_dir / "passage_directions.npy", directions)
        centers = np.stack([w.wall_center for w in walls])
        np.save(output_dir / "wall_centers.npy", centers)
        print(f"\n  Saved {len(walls)} passage directions to {output_dir}")

    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    print(f"\n다음 단계: passage_direction을 TopologicalAdapter에 주입")


if __name__ == "__main__":
    run()
