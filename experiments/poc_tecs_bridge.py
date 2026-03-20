"""
TECS Rust 엔진을 hidden states 분석에 직접 사용하는 브릿지.

ripser 대신 TECS의 PersistenceComputer를 사용.
tecs_rs Python 바인딩이 필요 (crates/tecs-python 빌드).
"""

import sys
import numpy as np
from pathlib import Path

# tecs_rs 바인딩 경로 추가
PYTHON_DIR = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(PYTHON_DIR))


def compute_ph_with_tecs(distance_matrix: np.ndarray, max_dim: int = 2) -> dict:
    """
    TECS Rust 엔진으로 PH 계산.

    Returns:
        {
            'beta0': int,
            'beta1': int,
            'beta2': int,
            'pairs': [(birth, death, dim, persistence), ...],
            'long_h1': [(birth, death), ...],
            'persistence_entropy': float,
        }
    """
    try:
        import tecs_rs

        # tecs_rs는 flat array를 받음
        n = distance_matrix.shape[0]
        flat = distance_matrix.flatten().tolist()

        result = tecs_rs.compute_topology_from_distance_matrix(flat, n, max_dim)
        return result

    except ImportError:
        print("[tecs_rs not available — build with: cd crates/tecs-python && maturin develop]")
        return None


def hidden_states_to_distance_matrix(
    hidden_states: np.ndarray,
    pca_dim: int = 50,
    max_points: int = 200,
) -> np.ndarray:
    """
    Hidden states (seq_len, hidden_dim) → 거리 행렬.

    1. FPS로 서브샘플
    2. PCA로 차원 축소
    3. 유클리드 거리 행렬
    """
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist, squareform

    points = hidden_states

    # FPS 서브샘플
    n = points.shape[0]
    if n > max_points:
        indices = [0]
        min_dists = np.full(n, np.inf)
        for _ in range(max_points - 1):
            last = indices[-1]
            dists = np.linalg.norm(points - points[last], axis=1)
            min_dists = np.minimum(min_dists, dists)
            indices.append(np.argmax(min_dists))
        points = points[indices]

    # PCA
    actual_dim = min(pca_dim, points.shape[0] - 1, points.shape[1])
    if actual_dim >= 2:
        pca = PCA(n_components=actual_dim)
        points = pca.fit_transform(points)

    # 거리 행렬
    return squareform(pdist(points, metric='euclidean'))


if __name__ == "__main__":
    # 테스트: 랜덤 점 구름으로 TECS 연동 확인
    print("Testing TECS bridge with random point cloud...")

    rng = np.random.default_rng(42)
    # 원 위의 점들 (β₁=1이 나와야 함)
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    noise = rng.normal(0, 0.05, points.shape)
    points += noise

    from scipy.spatial.distance import pdist, squareform
    dm = squareform(pdist(points))

    result = compute_ph_with_tecs(dm, max_dim=1)
    if result:
        print(f"TECS result: {result}")
    else:
        print("TECS not available. Falling back to ripser test...")
        try:
            from ripser import ripser
            r = ripser(dm, maxdim=1, distance_matrix=True)
            beta1 = sum(1 for b, d in r['dgms'][1] if d - b > 0.1)
            print(f"ripser β₁ = {beta1} (expected ≥ 1 for circle)")
        except ImportError:
            print("Neither tecs_rs nor ripser available. Install one:")
            print("  pip install ripser")
            print("  cd crates/tecs-python && maturin develop")
