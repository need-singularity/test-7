"""
GPU/모델 없이 빠르게 파이프라인 검증.

합성 데이터로 "hidden states처럼 생긴 점 구름"을 만들어서
PH 파이프라인이 β₁ hole을 잡는지 확인.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

def make_sphere_points(n=80, noise=0.05):
    """S² 위의 점들 — β₀=1, β₁=0, β₂=1"""
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2*np.pi, n)
    theta = np.arccos(rng.uniform(-1, 1, n))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    return points + rng.normal(0, noise, points.shape)

def make_torus_points(n=100, R=2.0, r=0.5, noise=0.05):
    """T² 위의 점들 — β₀=1, β₁=2, β₂=1"""
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2*np.pi, n)
    phi = rng.uniform(0, 2*np.pi, n)
    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    points = np.column_stack([x, y, z])
    return points + rng.normal(0, noise, points.shape)

def make_high_dim_with_hole(n=100, ambient_dim=100, hole_dim=2):
    """
    고차원 점 구름에 β₁ hole 임베딩.
    Llama hidden states를 흉내 냄: 대부분의 차원은 노이즈,
    특정 2차원 부분공간에 원(circle)이 숨어있음.
    """
    rng = np.random.default_rng(42)

    # 원 (hole의 원천)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])

    # 고차원으로 임베딩
    points = rng.normal(0, 0.1, (n, ambient_dim))
    points[:, :2] = circle  # 처음 2차원에 원 삽입

    return points

def run_test(name, points, expected_beta1):
    """단일 테스트 실행"""
    print(f"\n{'='*50}")
    print(f"Test: {name}")
    print(f"Shape: {points.shape}, Expected β₁ ≥ {expected_beta1}")
    print(f"{'='*50}")

    dm = squareform(pdist(points))

    try:
        from ripser import ripser
        result = ripser(dm, maxdim=2, distance_matrix=True)

        for dim in range(min(3, len(result['dgms']))):
            dgm = result['dgms'][dim]
            significant = [(b, d) for b, d in dgm if np.isfinite(d) and d - b > 0.05]
            print(f"  β_{dim}: {len(significant)} significant bars")
            if significant and dim == 1:
                for b, d in sorted(significant, key=lambda x: x[1]-x[0], reverse=True)[:3]:
                    print(f"    birth={b:.3f}, death={d:.3f}, persistence={d-b:.3f}")

        beta1 = sum(1 for b, d in result['dgms'][1] if np.isfinite(d) and d - b > 0.05)
        status = "PASS" if beta1 >= expected_beta1 else "FAIL"
        print(f"\n  Result: β₁={beta1}, {status}")
        return beta1 >= expected_beta1

    except ImportError:
        print("  [ripser not installed — pip install ripser]")
        return None


if __name__ == "__main__":
    results = []

    results.append(run_test("Sphere (S²)", make_sphere_points(), expected_beta1=0))
    results.append(run_test("Torus (T²)", make_torus_points(), expected_beta1=2))
    results.append(run_test(
        "High-dim with hidden hole (simulated LLM states)",
        make_high_dim_with_hole(n=100, ambient_dim=100),
        expected_beta1=1,
    ))
    results.append(run_test(
        "High-dim with hidden hole (4096-dim, like Llama)",
        make_high_dim_with_hole(n=80, ambient_dim=4096),
        expected_beta1=1,
    ))

    print(f"\n\n{'='*50}")
    print(f"Summary: {sum(1 for r in results if r)}/{len([r for r in results if r is not None])} passed")

    if all(r is None for r in results):
        print("\nripser가 필요합니다: pip install ripser")
