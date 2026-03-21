"""
T5 공정성 재검증
================

문제: 기존 T5 비교가 공정한가?

- Noise: randn(40, 50) × σ=1.0 → 점들이 50-dim 구에 퍼짐
- Structured: randn(40, 50) × 0.05 + 원(r=1.0) → 대부분 원점 근처, 일부만 원 위

이 두 point cloud는 **스케일과 기하학이 완전히 다름**.
Noise가 β₁이 더 많은 건 "PH artifact"가 아니라 "더 복잡한 기하학"일 수 있음.

공정한 비교:
  A. 같은 스케일의 noise vs structured
  B. Permutation test (structured의 좌표를 셔플)
  C. Matched-norm noise (같은 norm 분포를 가진 noise)
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy import stats
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

AMBIENT_DIM = 50
N_POINTS = 40
PERSISTENCE_FLOOR = 0.05
N_TRIALS = 50

RNG_MAP = np.random.RandomState(42)
WALL_DIMS = sorted(RNG_MAP.choice(AMBIENT_DIM, size=10, replace=False).tolist())

def measure(points, floor=PERSISTENCE_FLOOR):
    res = ripser(points, maxdim=1)
    dgm = res["dgms"][1]
    if len(dgm) == 0:
        return 0, 0.0, 0.0, []
    pers = dgm[:, 1] - dgm[:, 0]
    sig = pers[pers > floor]
    return len(sig), float(sig.max()) if len(sig) > 0 else 0.0, float(sig.sum()), sorted(pers.tolist(), reverse=True)

def make_structured(n_pts, rng):
    points = rng.randn(n_pts, AMBIENT_DIM) * 0.05
    pts_per_hole = n_pts // 2
    for h in range(2):
        plane_dims = rng.choice(WALL_DIMS, size=2, replace=False)
        i0, i1 = h * pts_per_hole, (h + 1) * pts_per_hole
        theta = np.linspace(0, 2 * np.pi, i1 - i0, endpoint=False)
        radius = 1.0 + 0.1 * rng.randn(len(theta))
        points[i0:i1, plane_dims[0]] = radius * np.cos(theta)
        points[i0:i1, plane_dims[1]] = radius * np.sin(theta)
    return points

print("=" * 70)
print("T5 FAIRNESS RE-VERIFICATION")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 0. 기존 비교의 불공정 확인
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 0. SCALE COMPARISON: Are the clouds comparable?")

scale_stats = {"noise": [], "structured": []}
for trial in range(N_TRIALS):
    rng = np.random.RandomState(500 + trial)
    n_pts = rng.randint(30, 44)

    noise = rng.randn(n_pts, AMBIENT_DIM)
    struct = make_structured(n_pts, rng)

    scale_stats["noise"].append({
        "mean_norm": np.linalg.norm(noise, axis=1).mean(),
        "mean_pdist": np.mean(pdist(noise)),
        "max_pdist": np.max(pdist(noise)),
    })
    scale_stats["structured"].append({
        "mean_norm": np.linalg.norm(struct, axis=1).mean(),
        "mean_pdist": np.mean(pdist(struct)),
        "max_pdist": np.max(pdist(struct)),
    })

print(f"\n  {'Metric':>20}  {'Noise':>12}  {'Structured':>12}  {'Ratio':>8}")
print(f"  {'-'*56}")
for key in ["mean_norm", "mean_pdist", "max_pdist"]:
    nv = np.mean([s[key] for s in scale_stats["noise"]])
    sv = np.mean([s[key] for s in scale_stats["structured"]])
    print(f"  {key:>20}  {nv:>12.4f}  {sv:>12.4f}  {nv/max(sv,1e-9):>8.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# A. SAME-SCALE NOISE — noise를 structured와 같은 스케일로
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("A. SAME-SCALE NOISE — match noise scale to structured")
print("=" * 70)

results_A = {"noise_scaled": [], "structured": [], "noise_raw": []}

for trial in range(N_TRIALS):
    rng = np.random.RandomState(500 + trial)
    n_pts = rng.randint(30, 44)

    struct = make_structured(n_pts, rng)
    struct_scale = np.std(struct)  # 전체 표준편차

    # Noise를 같은 스케일로 맞춤
    rng2 = np.random.RandomState(700 + trial)
    noise_scaled = rng2.randn(n_pts, AMBIENT_DIM) * struct_scale

    # Raw noise (기존)
    noise_raw = rng2.randn(n_pts, AMBIENT_DIM)

    b1_ns, mp_ns, tp_ns, _ = measure(noise_scaled)
    b1_s, mp_s, tp_s, _ = measure(struct)
    b1_nr, mp_nr, tp_nr, _ = measure(noise_raw)

    results_A["noise_scaled"].append({"b1": b1_ns, "mp": mp_ns, "tp": tp_ns})
    results_A["structured"].append({"b1": b1_s, "mp": mp_s, "tp": tp_s})
    results_A["noise_raw"].append({"b1": b1_nr, "mp": mp_nr, "tp": tp_nr})

print(f"\n  {'':>20}  {'Noise(raw)':>12}  {'Noise(scaled)':>14}  {'Structured':>12}")
print(f"  {'-'*62}")
for key, label in [("b1", "β₁ count"), ("mp", "max pers"), ("tp", "total pers")]:
    nr = np.mean([r[key] for r in results_A["noise_raw"]])
    ns = np.mean([r[key] for r in results_A["noise_scaled"]])
    sv = np.mean([r[key] for r in results_A["structured"]])
    print(f"  {label:>20}  {nr:>12.4f}  {ns:>14.4f}  {sv:>12.4f}")

ns_b1 = [r["b1"] for r in results_A["noise_scaled"]]
s_b1 = [r["b1"] for r in results_A["structured"]]
t_stat, p_val = stats.ttest_ind(ns_b1, s_b1)
u_stat, p_u = stats.mannwhitneyu(s_b1, ns_b1, alternative='two-sided')
print(f"\n  Noise(scaled) vs Structured β₁:")
print(f"    t-test: t={t_stat:.3f}, p={p_val:.6f}")
print(f"    Mann-Whitney: U={u_stat:.1f}, p={p_u:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# B. PERMUTATION TEST — structured의 좌표를 셔플하면?
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("B. PERMUTATION TEST — shuffle structured coordinates")
print("=" * 70)
print("  각 structured cloud에서 좌표를 행별(or 열별) 셔플.")
print("  구조를 파괴하되 marginal distribution은 보존.")

results_B = {"original": [], "row_shuffle": [], "col_shuffle": [], "full_shuffle": []}

for trial in range(N_TRIALS):
    rng = np.random.RandomState(500 + trial)
    n_pts = rng.randint(30, 44)
    struct = make_structured(n_pts, rng)

    b1_orig, mp_orig, _, _ = measure(struct)
    results_B["original"].append({"b1": b1_orig, "mp": mp_orig})

    # Row shuffle: 각 행(점)의 좌표를 독립적으로 셔플 → 구조 파괴
    rng3 = np.random.RandomState(900 + trial)
    row_shuf = struct.copy()
    for i in range(n_pts):
        rng3.shuffle(row_shuf[i])
    b1_rs, mp_rs, _, _ = measure(row_shuf)
    results_B["row_shuffle"].append({"b1": b1_rs, "mp": mp_rs})

    # Column shuffle: 각 열(차원)의 값을 독립적으로 셔플 → 점 간 관계 파괴
    col_shuf = struct.copy()
    for j in range(AMBIENT_DIM):
        rng3.shuffle(col_shuf[:, j])
    b1_cs, mp_cs, _, _ = measure(col_shuf)
    results_B["col_shuffle"].append({"b1": b1_cs, "mp": mp_cs})

    # Full shuffle: 전체 행렬을 flat으로 셔플 → 완전 파괴
    flat = struct.flatten()
    rng3.shuffle(flat)
    full_shuf = flat.reshape(struct.shape)
    b1_fs, mp_fs, _, _ = measure(full_shuf)
    results_B["full_shuffle"].append({"b1": b1_fs, "mp": mp_fs})

print(f"\n  {'':>15}  {'β₁ mean':>10}  {'β₁ std':>8}  {'max_pers':>10}")
print(f"  {'-'*48}")
for name in ["original", "row_shuffle", "col_shuffle", "full_shuffle"]:
    b1s = [r["b1"] for r in results_B[name]]
    mps = [r["mp"] for r in results_B[name]]
    print(f"  {name:>15}  {np.mean(b1s):>10.2f}  {np.std(b1s):>8.2f}  {np.mean(mps):>10.4f}")

# Permutation p-value: original β₁이 shuffled 대비 유의하게 다른가?
orig_b1s = [r["b1"] for r in results_B["original"]]
row_b1s = [r["b1"] for r in results_B["row_shuffle"]]
col_b1s = [r["b1"] for r in results_B["col_shuffle"]]

t_row, p_row = stats.ttest_ind(orig_b1s, row_b1s)
t_col, p_col = stats.ttest_ind(orig_b1s, col_b1s)
print(f"\n  Original vs Row-shuffle: t={t_row:.3f}, p={p_row:.6f}")
print(f"  Original vs Col-shuffle: t={t_col:.3f}, p={p_col:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# C. MATCHED-NORM NOISE
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("C. MATCHED-NORM NOISE — same per-point norm distribution")
print("=" * 70)

results_C = {"matched_noise": [], "structured": []}

for trial in range(N_TRIALS):
    rng = np.random.RandomState(500 + trial)
    n_pts = rng.randint(30, 44)
    struct = make_structured(n_pts, rng)

    # 각 점의 norm을 맞춘 noise
    struct_norms = np.linalg.norm(struct, axis=1)
    rng4 = np.random.RandomState(1100 + trial)
    noise = rng4.randn(n_pts, AMBIENT_DIM)
    noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
    matched_noise = noise / (noise_norms + 1e-10) * struct_norms[:, None]

    b1_mn, mp_mn, _, _ = measure(matched_noise)
    b1_s, mp_s, _, _ = measure(struct)

    results_C["matched_noise"].append({"b1": b1_mn, "mp": mp_mn})
    results_C["structured"].append({"b1": b1_s, "mp": mp_s})

mn_b1 = [r["b1"] for r in results_C["matched_noise"]]
s_b1 = [r["b1"] for r in results_C["structured"]]
mn_mp = [r["mp"] for r in results_C["matched_noise"]]
s_mp = [r["mp"] for r in results_C["structured"]]

print(f"\n  {'':>20}  {'Matched Noise':>14}  {'Structured':>12}")
print(f"  {'-'*50}")
print(f"  {'β₁ count mean':>20}  {np.mean(mn_b1):>14.2f}  {np.mean(s_b1):>12.2f}")
print(f"  {'β₁ count std':>20}  {np.std(mn_b1):>14.2f}  {np.std(s_b1):>12.2f}")
print(f"  {'max pers mean':>20}  {np.mean(mn_mp):>14.4f}  {np.mean(s_mp):>12.4f}")

t_mn, p_mn = stats.ttest_ind(mn_b1, s_b1)
u_mn, p_umn = stats.mannwhitneyu(mn_b1, s_b1, alternative='two-sided')
print(f"\n  Matched-norm noise vs Structured β₁:")
print(f"    t-test: t={t_mn:.3f}, p={p_mn:.6f}")
print(f"    Mann-Whitney: U={u_mn:.1f}, p={p_umn:.6f}")

# Max persistence comparison
t_mp2, p_mp2 = stats.ttest_ind(mn_mp, s_mp)
print(f"\n  Matched-norm noise vs Structured max_pers:")
print(f"    t-test: t={t_mp2:.3f}, p={p_mp2:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# D. NULL MODEL — structured cloud에서 β₁의 통계적 유의성
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("D. NULL MODEL — Is each structured β₁ statistically significant?")
print("=" * 70)
print("  각 structured cloud에 대해 같은 marginal로 null cloud 100개 생성")
print("  실제 β₁이 null 분포의 몇 %ile에 위치하는지 확인")

n_null = 100
significant_count = 0
total_count = 0

print(f"\n  {'Trial':>6}  {'n_pts':>6}  {'β₁':>4}  {'null_mean':>10}  {'null_std':>9}  {'percentile':>11}  {'p<0.05?':>8}")
print(f"  {'-'*60}")

for trial in range(min(N_TRIALS, 20)):  # 20 trials for speed
    rng = np.random.RandomState(500 + trial)
    n_pts = rng.randint(30, 44)
    struct = make_structured(n_pts, rng)
    b1_real, mp_real, _, _ = measure(struct)

    # Null: col-shuffle (marginal 보존, 구조 파괴)
    null_b1s = []
    for null_trial in range(n_null):
        rng_null = np.random.RandomState(2000 + trial * 1000 + null_trial)
        null_cloud = struct.copy()
        for j in range(AMBIENT_DIM):
            rng_null.shuffle(null_cloud[:, j])
        b1_null, _, _, _ = measure(null_cloud)
        null_b1s.append(b1_null)

    null_mean = np.mean(null_b1s)
    null_std = np.std(null_b1s)
    # percentile: 실제 β₁이 null에서 얼마나 극단인가
    pctile = np.mean([n <= b1_real for n in null_b1s]) * 100
    # p-value (one-sided: real이 null보다 큰가?)
    if null_std > 0:
        z = (b1_real - null_mean) / null_std
    else:
        z = 0 if b1_real == null_mean else float('inf')
    p_sig = 1 - stats.norm.cdf(z) if z != float('inf') else 0.0
    sig = "YES" if p_sig < 0.05 else "no"

    total_count += 1
    if p_sig < 0.05:
        significant_count += 1

    print(f"  {trial:>6}  {n_pts:>6}  {b1_real:>4}  {null_mean:>10.2f}  {null_std:>9.2f}  {pctile:>10.0f}%  {sig:>8}")

print(f"\n  Significant (p<0.05): {significant_count}/{total_count} ({significant_count/max(total_count,1)*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("T5 FAIRNESS — FINAL VERDICT")
print("=" * 70)

print(f"""
  기존 T5 비교의 공정성 문제:
  ─────────────────────────
  기존: noise(σ=1.0) vs structured(σ=0.05+circle)
  → 스케일이 {np.mean([s['mean_norm'] for s in scale_stats['noise']]):.1f}x 차이
  → noise의 β₁이 더 많은 건 '더 넓게 퍼진 점들' 때문일 수 있음

  공정한 비교 결과:
  ─────────────────
  A. Same-scale noise β₁:  {np.mean(ns_b1):.2f}  vs  Structured: {np.mean(s_b1):.2f}
  B. Row-shuffle β₁:       {np.mean(row_b1s):.2f}  vs  Original:   {np.mean(orig_b1s):.2f}
  C. Matched-norm noise β₁: {np.mean(mn_b1):.2f}  vs  Structured: {np.mean([r['b1'] for r in results_C['structured']]):.2f}
  D. Null model significant: {significant_count}/{total_count}

  결론:
""")

# Determine verdict based on results
noise_scaled_mean = np.mean(ns_b1)
struct_mean = np.mean(s_b1)
matched_mean = np.mean(mn_b1)

if struct_mean > noise_scaled_mean * 1.2 or significant_count > total_count * 0.5:
    print(f"  → 공정한 비교에서 structured β₁이 유의하게 다름")
    print(f"  → 기존 T5의 '심각한 문제'는 불공정 비교 때문이었을 수 있음")
    print(f"  → PH artifact 우려가 완화됨")
elif abs(struct_mean - noise_scaled_mean) < 0.5 and significant_count < total_count * 0.3:
    print(f"  → 공정한 비교에서도 structured와 noise가 구분 불가")
    print(f"  → T5 문제는 실재함 — PH artifact 위험 유지")
else:
    print(f"  → 부분적 구분 가능 — 추가 검증 필요")
    print(f"  → β₁ count보다 persistence diagram 형태나 다른 지표가 필요할 수 있음")
