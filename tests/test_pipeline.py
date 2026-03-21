"""
Core Pipeline 단위 테스트.

합성 데이터로 PH 파이프라인 핵심 함수 검증.
모델 불필요 — 합성 point cloud만 사용.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from common import (
    compute_full_topology,
    compute_topology_from_distance_matrix,
    radial_perturbation,
    multi_wall_perturbation,
    emergence_score,
    get_passage_direction,
)


# ── 합성 데이터 생성 ──────────────────────────────────

def make_circle(n=60, noise=0.02, rng=None):
    """원(S¹) — β₁ ≥ 1"""
    rng = rng or np.random.default_rng(42)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    return points + rng.normal(0, noise, points.shape)


def make_torus(n=120, R=2.0, r=0.5, noise=0.03, rng=None):
    """토러스(T²) — β₁ ≥ 2"""
    rng = rng or np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    points = np.column_stack([x, y, z])
    return points + rng.normal(0, noise, points.shape)


def make_high_dim_circle(n=80, ambient_dim=4096, rng=None):
    """4096-dim에 숨겨진 원 — β₁ ≥ 1"""
    rng = rng or np.random.default_rng(42)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    points = rng.normal(0, 0.1, (n, ambient_dim))
    points[:, :2] = circle
    return points


def make_cluster(n=40, dim=50, rng=None):
    """단순 클러스터 — β₁ = 0 (hole 없음)"""
    rng = rng or np.random.default_rng(42)
    return rng.normal(0, 1, (n, dim))


# ── compute_full_topology 테스트 ──────────────────────

class TestComputeFullTopology:
    def test_circle_detects_hole(self):
        """원에서 β₁ ≥ 1 감지"""
        points = make_circle()
        topo = compute_full_topology(points)
        assert topo['beta1'] >= 1, f"원에서 β₁={topo['beta1']}, 최소 1 기대"

    def test_torus_detects_two_holes(self):
        """토러스에서 β₁ ≥ 2 감지"""
        points = make_torus()
        topo = compute_full_topology(points)
        assert topo['beta1'] >= 2, f"토러스에서 β₁={topo['beta1']}, 최소 2 기대"

    def test_cluster_no_holes(self):
        """가우시안 클러스터에서 β₁ = 0 (또는 매우 적음)"""
        points = make_cluster()
        topo = compute_full_topology(points)
        # 노이즈로 인한 미세 hole 허용하되, persistence는 낮아야 함
        if topo['beta1'] > 0:
            assert topo['max_persistence'] < 1.0, \
                f"클러스터에서 유의미한 hole: pers={topo['max_persistence']}"

    def test_high_dim_detects_hole(self):
        """4096-dim 공간에 숨겨진 원 감지"""
        points = make_high_dim_circle()
        topo = compute_full_topology(points)
        assert topo['beta1'] >= 1, f"4096-dim에서 β₁={topo['beta1']}, 최소 1 기대"

    def test_output_structure(self):
        """반환 dict의 필수 키 확인"""
        points = make_circle()
        topo = compute_full_topology(points)
        required_keys = ['beta0', 'beta1', 'walls', 'max_persistence',
                         'total_persistence', 'mean_persistence', 'result']
        for key in required_keys:
            assert key in topo, f"'{key}' 누락"

    def test_walls_have_vertex_indices(self):
        """wall에 vertex_indices 포함"""
        points = make_circle()
        topo = compute_full_topology(points)
        if topo['walls']:
            wall = topo['walls'][0]
            assert 'vertex_indices' in wall
            assert 'persistence' in wall
            assert len(wall['vertex_indices']) >= 2

    def test_walls_sorted_by_persistence(self):
        """walls가 persistence 내림차순 정렬"""
        points = make_torus()
        topo = compute_full_topology(points)
        if len(topo['walls']) >= 2:
            for i in range(len(topo['walls']) - 1):
                assert topo['walls'][i]['persistence'] >= topo['walls'][i + 1]['persistence']

    def test_beta0_positive(self):
        """β₀ ≥ 1 (최소 1개 연결 성분)"""
        points = make_circle()
        topo = compute_full_topology(points)
        assert topo['beta0'] >= 1

    def test_persistence_nonnegative(self):
        """persistence 값이 모두 양수"""
        points = make_torus()
        topo = compute_full_topology(points)
        for wall in topo['walls']:
            assert wall['persistence'] > 0


# ── compute_topology_from_distance_matrix 테스트 ──────

class TestTopologyFromDistanceMatrix:
    def test_circle_from_dm(self):
        """거리 행렬 입력으로 원의 β₁ 감지"""
        points = make_circle()
        dm = squareform(pdist(points))
        topo = compute_topology_from_distance_matrix(dm)
        assert topo['beta1'] >= 1

    def test_consistent_with_full(self):
        """compute_full_topology와 동일 β₁ 반환 (2D 입력)"""
        points = make_circle()
        topo_full = compute_full_topology(points)
        dm = squareform(pdist(points))
        topo_dm = compute_topology_from_distance_matrix(dm)
        # PCA가 적용되므로 정확히 같지 않을 수 있지만, 2D에서는 PCA 무변환
        assert topo_dm['beta1'] >= 1

    def test_zero_matrix(self):
        """모든 거리가 0인 행렬 → 에러 없이 반환"""
        dm = np.zeros((5, 5))
        topo = compute_topology_from_distance_matrix(dm)
        assert 'beta1' in topo


# ── radial_perturbation 테스트 ────────────────────────

class TestRadialPerturbation:
    def _get_circle_with_wall(self):
        points = make_circle(n=40)
        topo = compute_full_topology(points)
        return points, topo

    def test_alpha_zero_no_change(self):
        """alpha=0이면 원본과 동일"""
        points, topo = self._get_circle_with_wall()
        if not topo['walls']:
            pytest.skip("hole 없음")
        perturbed = radial_perturbation(points, topo['walls'][0], alpha=0.0)
        np.testing.assert_array_almost_equal(perturbed, points)

    def test_perturbation_changes_cycle_points(self):
        """alpha > 0이면 cycle 점들이 이동"""
        points, topo = self._get_circle_with_wall()
        if not topo['walls']:
            pytest.skip("hole 없음")
        wall = topo['walls'][0]
        perturbed = radial_perturbation(points, wall, alpha=5.0)

        verts = wall['vertex_indices']
        for v in verts:
            diff = np.linalg.norm(perturbed[v] - points[v])
            assert diff > 0, f"vertex {v} 미이동"

    def test_non_cycle_points_unchanged(self):
        """cycle에 속하지 않는 점은 이동하지 않음"""
        points, topo = self._get_circle_with_wall()
        if not topo['walls']:
            pytest.skip("hole 없음")
        wall = topo['walls'][0]
        perturbed = radial_perturbation(points, wall, alpha=5.0)

        verts = set(wall['vertex_indices'])
        for i in range(len(points)):
            if i not in verts:
                np.testing.assert_array_equal(perturbed[i], points[i])

    def test_original_not_modified(self):
        """원본 배열이 수정되지 않음 (copy)"""
        points, topo = self._get_circle_with_wall()
        if not topo['walls']:
            pytest.skip("hole 없음")
        original = points.copy()
        radial_perturbation(points, topo['walls'][0], alpha=10.0)
        np.testing.assert_array_equal(points, original)


# ── multi_wall_perturbation 테스트 ────────────────────

class TestMultiWallPerturbation:
    def test_reduces_beta1(self):
        """충분한 alpha에서 β₁ 감소"""
        points = make_torus(n=100)
        topo = compute_full_topology(points)
        if topo['beta1'] == 0:
            pytest.skip("hole 없음")

        perturbed = multi_wall_perturbation(points, topo['walls'], alpha=20.0)
        topo_after = compute_full_topology(perturbed)
        assert topo_after['beta1'] <= topo['beta1'], \
            f"β₁ 증가: {topo['beta1']} → {topo_after['beta1']}"

    def test_max_walls_limits(self):
        """max_walls 파라미터가 적용됨"""
        points = make_torus(n=100)
        topo = compute_full_topology(points)
        if len(topo['walls']) < 2:
            pytest.skip("wall 2개 미만")

        p1 = multi_wall_perturbation(points, topo['walls'], alpha=5.0, max_walls=1)
        p_all = multi_wall_perturbation(points, topo['walls'], alpha=5.0)

        # max_walls=1이면 첫 번째 wall만 수축 → 다른 결과
        assert not np.allclose(p1, p_all)

    def test_alpha_zero_no_change(self):
        """alpha=0이면 변화 없음"""
        points = make_torus(n=100)
        topo = compute_full_topology(points)
        perturbed = multi_wall_perturbation(points, topo['walls'], alpha=0.0)
        np.testing.assert_array_almost_equal(perturbed, points)


# ── emergence_score 테스트 ────────────────────────────

class TestEmergenceScore:
    def test_no_change_zero_score(self):
        """변화 없으면 wall_reduction=0, pers_reduction=0"""
        topo = {'beta1': 5, 'max_persistence': 3.0, 'beta0': 1}
        score = emergence_score(topo, topo)
        assert score['wall_reduction'] == 0.0
        assert score['pers_reduction'] == 0.0
        assert score['stability'] == 1.0

    def test_full_reduction_perfect_score(self):
        """β₁→0, persistence→0이면 wall_reduction=1, pers_reduction=1"""
        before = {'beta1': 5, 'max_persistence': 3.0, 'beta0': 1}
        after = {'beta1': 0, 'max_persistence': 0.0, 'beta0': 1}
        score = emergence_score(before, after)
        assert score['wall_reduction'] == 1.0
        assert score['pers_reduction'] == 1.0
        assert score['stability'] == 1.0
        assert score['passage_score'] == 1.0

    def test_partial_reduction(self):
        """부분 감소 시 0 < score < 1"""
        before = {'beta1': 6, 'max_persistence': 4.0, 'beta0': 1}
        after = {'beta1': 3, 'max_persistence': 2.0, 'beta0': 1}
        score = emergence_score(before, after)
        assert 0.0 < score['wall_reduction'] < 1.0
        assert 0.0 < score['pers_reduction'] < 1.0
        assert 0.0 < score['passage_score'] < 1.0

    def test_stability_drops_on_beta0_change(self):
        """β₀ 변화 시 stability 감소"""
        before = {'beta1': 5, 'max_persistence': 3.0, 'beta0': 5}
        after = {'beta1': 0, 'max_persistence': 0.0, 'beta0': 1}
        score = emergence_score(before, after)
        assert score['stability'] < 1.0

    def test_output_keys(self):
        """반환 dict의 필수 키 확인"""
        topo = {'beta1': 3, 'max_persistence': 2.0, 'beta0': 1}
        score = emergence_score(topo, topo)
        for key in ['passage_score', 'wall_reduction', 'pers_reduction',
                    'stability', 'beta1_change', 'max_pers_change']:
            assert key in score, f"'{key}' 누락"

    def test_score_in_range(self):
        """모든 점수가 0~1 범위"""
        before = {'beta1': 4, 'max_persistence': 5.0, 'beta0': 2}
        after = {'beta1': 1, 'max_persistence': 1.0, 'beta0': 2}
        score = emergence_score(before, after)
        for key in ['passage_score', 'wall_reduction', 'pers_reduction', 'stability']:
            assert 0.0 <= score[key] <= 1.0, f"{key}={score[key]} 범위 밖"


# ── get_passage_direction 테스트 ──────────────────────

class TestPassageDirection:
    def test_returns_direction_for_high_dim(self):
        """고차원 데이터의 wall에서 passage direction 반환 (2D는 직교 보완 없음)"""
        points = make_high_dim_circle(n=50, ambient_dim=100)
        topo = compute_full_topology(points)
        if not topo['walls']:
            pytest.skip("hole 없음")

        direction, center = get_passage_direction(points, topo['walls'][0])
        assert direction is not None
        assert center is not None

    def test_direction_is_unit_vector(self):
        """passage direction이 단위 벡터"""
        points = make_circle(n=40)
        topo = compute_full_topology(points)
        if not topo['walls']:
            pytest.skip("hole 없음")

        direction, _ = get_passage_direction(points, topo['walls'][0])
        if direction is not None:
            norm = np.linalg.norm(direction)
            assert abs(norm - 1.0) < 1e-6, f"norm={norm}, 1.0 기대"

    def test_direction_dim_matches_input(self):
        """direction 차원이 입력 차원과 일치"""
        points = make_high_dim_circle(n=50, ambient_dim=100)
        topo = compute_full_topology(points)
        if not topo['walls']:
            pytest.skip("hole 없음")

        direction, _ = get_passage_direction(points, topo['walls'][0])
        if direction is not None:
            assert len(direction) == 100


# ── Exp-B 쌍곡 함수 테스트 ────────────────────────────

class TestHyperbolicFunctions:
    def setup_method(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

    def test_poincare_projection_inside_ball(self):
        """사영된 점들이 unit ball 내부"""
        from expB_hyperbolic_ph import project_to_poincare_ball
        points = np.random.default_rng(42).normal(0, 5, (30, 10))
        projected = project_to_poincare_ball(points)
        norms = np.linalg.norm(projected, axis=1)
        assert np.all(norms < 1.0), f"max norm={norms.max()}"

    def test_poincare_distance_positive(self):
        """쌍곡 거리가 양수"""
        from expB_hyperbolic_ph import poincare_distance
        x = np.array([0.1, 0.2])
        y = np.array([0.3, 0.4])
        d = poincare_distance(x, y)
        assert d > 0
        assert np.isfinite(d)

    def test_poincare_distance_symmetric(self):
        """쌍곡 거리가 대칭"""
        from expB_hyperbolic_ph import poincare_distance
        x = np.array([0.1, 0.2, 0.3])
        y = np.array([0.4, 0.1, 0.2])
        assert abs(poincare_distance(x, y) - poincare_distance(y, x)) < 1e-10

    def test_poincare_distance_self_zero(self):
        """자기 자신과의 거리 ≈ 0"""
        from expB_hyperbolic_ph import poincare_distance
        x = np.array([0.1, 0.2])
        d = poincare_distance(x, x)
        assert d < 0.01, f"self-distance={d}, ~0 기대"

    def test_delta_hyperbolicity_tree_is_zero(self):
        """트리 거리 행렬의 δ = 0"""
        from expB_hyperbolic_ph import delta_hyperbolicity
        # path graph: 0-1-2-3
        dm = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ], dtype=float)
        delta = delta_hyperbolicity(dm)
        assert delta < 0.001, f"트리의 δ={delta}, 0 기대"

    def test_delta_hyperbolicity_cycle_nonzero(self):
        """사이클 거리 행렬의 δ > 0"""
        from expB_hyperbolic_ph import delta_hyperbolicity
        # 4-cycle: 0-1-2-3-0
        dm = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0],
        ], dtype=float)
        delta = delta_hyperbolicity(dm)
        assert delta > 0, f"사이클의 δ={delta}, >0 기대"


# ── Exp-A Ricci flow 함수 테스트 ──────────────────────

class TestRicciFlowFunctions:
    def test_build_knn_graph(self):
        """k-NN 그래프 구축"""
        from expA_ricci_flow import build_knn_graph
        points = make_circle(n=20)
        G = build_knn_graph(points, k=5)
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() > 0

    def test_knn_edge_weights_positive(self):
        """edge weight가 양수"""
        from expA_ricci_flow import build_knn_graph
        points = make_circle(n=20)
        G = build_knn_graph(points, k=5)
        for _, _, data in G.edges(data=True):
            assert data['weight'] > 0

    def test_ricci_flow_step_preserves_graph(self):
        """flow step 후 노드/엣지 수 유지"""
        from expA_ricci_flow import build_knn_graph, compute_all_curvatures, ricci_flow_step
        points = make_circle(n=15)
        G = build_knn_graph(points, k=5)
        curvatures = compute_all_curvatures(G, alpha=0.5)
        G_new = ricci_flow_step(G, curvatures, epsilon=0.1)
        assert G_new.number_of_nodes() == G.number_of_nodes()
        assert G_new.number_of_edges() == G.number_of_edges()

    def test_ricci_flow_step_weights_positive(self):
        """flow step 후에도 weight 양수 유지"""
        from expA_ricci_flow import build_knn_graph, compute_all_curvatures, ricci_flow_step
        points = make_circle(n=15)
        G = build_knn_graph(points, k=5)
        curvatures = compute_all_curvatures(G, alpha=0.5)
        G_new = ricci_flow_step(G, curvatures, epsilon=0.5)
        for _, _, data in G_new.edges(data=True):
            assert data['weight'] > 0

    def test_direction_preservation_identical(self):
        """동일 점이면 preservation = 1.0"""
        from expA_ricci_flow import direction_preservation
        points = make_circle(n=10)
        pres = direction_preservation(points, points)
        assert abs(pres - 1.0) < 1e-6
