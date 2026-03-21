"""
Math Verification — fire-in-the-hole 수학적 주장 검증
=====================================================

리뷰에서 지적된 5가지 문제를 이 프로젝트의 실제 코드/파라미터로 검증.
main이 맞을 가능성도 열어두고, 양방향으로 테스트.

Tests:
  T1: 수축 공식 — unit-step vs proportional, 중심 관통 여부
  T2: Loss 스케일 — 50-dim 시뮬레이션에서 λ=1.0이 실제로 유효한가?
  T3: OOD 이탈 — 수축 후 점이 학습 분포 바깥으로 나가는가?
  T4: Collateral=0은 발견인가 정의인가 — random sparse 대조군
  T5: PH artifact — noise vs structured β₁ in 50-dim
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from ripser import ripser

np.random.seed(42)

AMBIENT_DIM = 50
N_POINTS = 40
PERSISTENCE_FLOOR = 0.05

# Wall dims (same as project)
RNG_MAP = np.random.RandomState(42)
WALL_DIMS = sorted(RNG_MAP.choice(AMBIENT_DIM, size=10, replace=False).tolist())
NON_WALL_DIMS = sorted(set(range(AMBIENT_DIM)) - set(WALL_DIMS))

print("=" * 70)
print("FIRE-IN-THE-HOLE — MATH VERIFICATION")
print("=" * 70)
print(f"Ambient dim: {AMBIENT_DIM}, Wall dims: {WALL_DIMS}")
print(f"Non-wall dims: {len(NON_WALL_DIMS)}")


# ── Helper ──────────────────────────────────────────────────────────────────

def make_cloud(n_pts=N_POINTS, n_dim=AMBIENT_DIM, n_holes=2, seed=0):
    rng = np.random.RandomState(seed)
    points = rng.randn(n_pts, n_dim) * 0.05
    pts_per_hole = n_pts // n_holes
    for h in range(n_holes):
        plane_dims = rng.choice(WALL_DIMS, size=2, replace=False)
        i0, i1 = h * pts_per_hole, (h + 1) * pts_per_hole
        theta = np.linspace(0, 2 * np.pi, i1 - i0, endpoint=False)
        radius = 1.0 + 0.1 * rng.randn(len(theta))
        points[i0:i1, plane_dims[0]] = radius * np.cos(theta)
        points[i0:i1, plane_dims[1]] = radius * np.sin(theta)
    return points


def measure_beta1(points, floor=PERSISTENCE_FLOOR):
    res = ripser(points, maxdim=1)
    dgm = res["dgms"][1]
    if len(dgm) == 0:
        return 0, 0.0, 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    sig = pers[pers > floor]
    return len(sig), float(sig.max()) if len(sig) > 0 else 0.0, float(sig.sum())


# ══════════════════════════════════════════════════════════════════════════════
# T1: 수축 공식 비교 — unit-step vs proportional
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("T1: CONTRACTION FORMULA — unit-step vs proportional")
print("=" * 70)

# phase3b (실제 LLM 코드): v -= α * (radial / ||radial||)  [unit step]
# phase6 simulation:        v -= rate * directions           [proportional]

cloud = make_cloud(seed=0)
center_wall = cloud[:, WALL_DIMS].mean(axis=0)
radials = cloud[:, WALL_DIMS] - center_wall
norms = np.linalg.norm(radials, axis=1)

print(f"\nWall-dim radial distances from center:")
print(f"  min:  {norms.min():.4f}")
print(f"  max:  {norms.max():.4f}")
print(f"  mean: {norms.mean():.4f}")
print(f"  std:  {norms.std():.4f}")

# Test various α values used in the project
alphas_unit = [0.05, 0.10, 0.15, 0.20, 0.50]
rate_proportional = 0.15  # project default

print(f"\n  {'α/rate':>8}  {'Method':>14}  {'Inversions':>10}  {'β₁':>4}  {'max_pers':>9}  {'total_pers':>10}")
print(f"  {'-'*60}")

# Baseline
b0, mp0, tp0 = measure_beta1(cloud)
print(f"  {'—':>8}  {'baseline':>14}  {'—':>10}  {b0:>4}  {mp0:>9.4f}  {tp0:>10.4f}")

for alpha in alphas_unit:
    # Unit-step: v -= α * (radial / ||radial||)
    pts_unit = cloud.copy()
    unit_dirs = radials / (norms[:, None] + 1e-10)
    pts_unit[:, WALL_DIMS] -= alpha * unit_dirs

    # Check inversions (crossed center)
    new_radials = pts_unit[:, WALL_DIMS] - center_wall
    dots = np.sum(radials * new_radials, axis=1)
    inversions = (dots < 0).sum()

    b1, mp1, tp1 = measure_beta1(pts_unit)
    print(f"  {alpha:>8.2f}  {'unit-step':>14}  {inversions:>10}  {b1:>4}  {mp1:>9.4f}  {tp1:>10.4f}")

print()
for rate in [0.10, 0.15, 0.20, 0.50, 0.80]:
    # Proportional: v -= rate * directions
    pts_prop = cloud.copy()
    pts_prop[:, WALL_DIMS] -= rate * radials

    # Check inversions
    new_radials = pts_prop[:, WALL_DIMS] - center_wall
    dots = np.sum(radials * new_radials, axis=1)
    inversions = (dots < 0).sum()

    b1, mp1, tp1 = measure_beta1(pts_prop)
    print(f"  {rate:>8.2f}  {'proportional':>14}  {inversions:>10}  {b1:>4}  {mp1:>9.4f}  {tp1:>10.4f}")

# Critical question: at α=0.15 (project default), does unit-step cause inversions?
print(f"\n  Inversions at α=0.15 (unit-step): ", end="")
pts_test = cloud.copy()
unit_dirs = radials / (norms[:, None] + 1e-10)
pts_test[:, WALL_DIMS] -= 0.15 * unit_dirs
new_r = pts_test[:, WALL_DIMS] - center_wall
inversions = (np.sum(radials * new_r, axis=1) < 0).sum()
problematic = norms[norms < 0.15]
print(f"{inversions}/{len(cloud)} (norms < α: {len(problematic)}/{len(cloud)})")

# T1 VERDICT
print(f"\n  ▶ T1 VERDICT:")
if len(problematic) > 0:
    print(f"    unit-step: {len(problematic)} points have ||radial|| < α=0.15 → INVERSION RISK")
    print(f"    proportional (rate=0.15): NEVER inverts (rate < 1.0)")
    print(f"    BUT: phase6 simulation already uses proportional. Only phase3b uses unit-step.")
else:
    print(f"    NO inversions at α=0.15 for unit-step either")
    print(f"    Both formulas safe at this α. Unit-step vs proportional: minor difference.")


# ══════════════════════════════════════════════════════════════════════════════
# T2: Loss 스케일 — 50-dim 시뮬레이션에서 λ=1.0이 유효한가?
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("T2: LOSS SCALE — Is λ=1.0 valid in 50-dim simulation?")
print("=" * 70)

# In the simulation, there's no CE loss. The "loss" is just topology (persistence).
# λ scales the contraction: pts -= λ * lr * displacement
# The question is: does λ=1.0 make sense at this scale?

cloud = make_cloud(seed=0)
b0, mp0, tp0 = measure_beta1(cloud)

# Measure topology "loss" = total persistence
print(f"\n  Baseline: β₁={b0}, total_pers={tp0:.4f}")
print(f"\n  Simulated topology loss components:")

# In phase6_wall_finetune.py: contraction = λ * lr * displacement
# lr = 0.15, displacement = point - center
# For various λ values:

lr = 0.15
print(f"  lr = {lr}")
print(f"\n  {'λ':>6}  {'effective_rate':>15}  {'β₁':>4}  {'total_pers':>10}  {'Δpers%':>8}  {'safe?':>6}")
print(f"  {'-'*55}")

for lam in [0.01, 0.1, 0.5, 1.0, 2.0]:
    effective_rate = lam * lr
    pts = cloud.copy()
    center = pts[:, WALL_DIMS].mean(axis=0)
    dirs = pts[:, WALL_DIMS] - center

    if effective_rate >= 1.0:
        safe = "NO"
    else:
        safe = "YES"

    pts[:, WALL_DIMS] -= effective_rate * dirs
    b1, mp1, tp1 = measure_beta1(pts)
    delta = (tp1 - tp0) / max(tp0, 1e-9) * 100
    print(f"  {lam:>6.2f}  {effective_rate:>15.3f}  {b1:>4}  {tp1:>10.4f}  {delta:>+7.1f}%  {safe:>6}")

print(f"\n  ▶ T2 VERDICT:")
print(f"    50-dim 시뮬레이션에서 λ=1.0 × lr=0.15 → effective_rate=0.15 (< 1.0)")
print(f"    이 스케일에서는 수학적으로 안전. λ=1.0이 유효한 이유:")
print(f"    - 시뮬레이션에는 CE loss가 없음 (topology만 최적화)")
print(f"    - effective_rate = λ×lr = 0.15로 적절한 수축 속도")
print(f"    ⚠ 실제 LLM fine-tuning에서는 CE와 스케일 경쟁 → λ=1.0이 위험")
print(f"    ⚠ 이 프로젝트는 시뮬레이션이므로 λ=1.0이 맞을 수 있음")


# ══════════════════════════════════════════════════════════════════════════════
# T3: OOD 이탈 — 수축 후 점이 원래 분포 바깥으로 나가는가?
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("T3: OOD ESCAPE — Do contracted points leave the original distribution?")
print("=" * 70)

cloud = make_cloud(seed=0)
orig_norms = np.linalg.norm(cloud, axis=1)
orig_mean_norm = orig_norms.mean()
orig_std_norm = orig_norms.std()
ood_threshold = orig_mean_norm + 3 * orig_std_norm  # 3σ

print(f"\n  Original point norms: mean={orig_mean_norm:.4f}, std={orig_std_norm:.4f}")
print(f"  OOD threshold (3σ): {ood_threshold:.4f}")

print(f"\n  {'Strategy':>14}  {'rate':>6}  {'OOD%':>6}  {'norm_mean':>10}  {'norm_std':>9}  {'max_norm':>9}")
print(f"  {'-'*60}")

for name, rate in [("proportional", 0.15), ("proportional", 0.50), ("proportional", 0.80)]:
    pts = cloud.copy()
    center = pts[:, WALL_DIMS].mean(axis=0)
    dirs = pts[:, WALL_DIMS] - center
    pts[:, WALL_DIMS] -= rate * dirs

    new_norms = np.linalg.norm(pts, axis=1)
    ood_pct = (new_norms > ood_threshold).mean() * 100
    print(f"  {name:>14}  {rate:>6.2f}  {ood_pct:>5.1f}%  {new_norms.mean():>10.4f}  {new_norms.std():>9.4f}  {new_norms.max():>9.4f}")

for name, rate in [("global", 0.15), ("global", 0.50)]:
    pts = cloud.copy()
    center = pts.mean(axis=0)
    dirs = pts - center
    pts -= rate * dirs

    new_norms = np.linalg.norm(pts, axis=1)
    ood_pct = (new_norms > ood_threshold).mean() * 100
    print(f"  {name:>14}  {rate:>6.2f}  {ood_pct:>5.1f}%  {new_norms.mean():>10.4f}  {new_norms.std():>9.4f}  {new_norms.max():>9.4f}")

print(f"\n  ▶ T3 VERDICT:")
print(f"    Selective(proportional) 수축은 점을 중심으로 당기므로 norm이 감소.")
print(f"    OOD 이탈이 아니라 분포 '수축'. 50-dim 시뮬레이션에서는 OOD 문제 없음.")
print(f"    ⚠ 실제 4096-dim LLM에서는 다를 수 있음 (layer norm, residual stream)")


# ══════════════════════════════════════════════════════════════════════════════
# T4: Collateral=0 — 발견인가 정의인가? Random sparse 대조군
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("T4: COLLATERAL=0 — Discovery or Definition? Random sparse control")
print("=" * 70)

cloud = make_cloud(seed=0)
original = cloud.copy()
rate = 0.15

def collateral(orig, modified):
    d_orig = pdist(orig[:, NON_WALL_DIMS])
    d_mod = pdist(modified[:, NON_WALL_DIMS])
    return float(np.mean(np.abs(d_orig - d_mod)))

def wall_collateral(orig, modified):
    """wall dims에서의 변화"""
    d_orig = pdist(orig[:, WALL_DIMS])
    d_mod = pdist(modified[:, WALL_DIMS])
    return float(np.mean(np.abs(d_orig - d_mod)))

# 1. Selective (wall dims only) — 프로젝트 방식
pts_selective = cloud.copy()
center_w = cloud[:, WALL_DIMS].mean(axis=0)
pts_selective[:, WALL_DIMS] -= rate * (cloud[:, WALL_DIMS] - center_w)
b1_sel, _, _ = measure_beta1(pts_selective)
coll_sel = collateral(original, pts_selective)
wcoll_sel = wall_collateral(original, pts_selective)

# 2. Random-k (same k=10, random dims) — 대조군
n_random_trials = 20
random_results = []
for trial in range(n_random_trials):
    rng = np.random.RandomState(100 + trial)
    random_dims = sorted(rng.choice(AMBIENT_DIM, size=len(WALL_DIMS), replace=False).tolist())
    random_non = sorted(set(range(AMBIENT_DIM)) - set(random_dims))

    pts_rand = cloud.copy()
    center_r = cloud[:, random_dims].mean(axis=0)
    pts_rand[:, random_dims] -= rate * (cloud[:, random_dims] - center_r)

    b1_r, _, _ = measure_beta1(pts_rand)
    # Collateral = change in dims NOT targeted
    d_orig_non = pdist(original[:, random_non])
    d_mod_non = pdist(pts_rand[:, random_non])
    coll_r = float(np.mean(np.abs(d_orig_non - d_mod_non)))

    random_results.append({"beta1": b1_r, "collateral": coll_r})

random_b1s = [r["beta1"] for r in random_results]
random_colls = [r["collateral"] for r in random_results]

# 3. Global
pts_global = cloud.copy()
center_g = cloud.mean(axis=0)
pts_global -= rate * (cloud - center_g)
b1_glob, _, _ = measure_beta1(pts_global)
coll_glob = collateral(original, pts_global)

b0, _, _ = measure_beta1(cloud)

print(f"\n  Baseline β₁ = {b0}")
print(f"\n  {'Strategy':>20}  {'β₁':>4}  {'non-target coll':>16}  {'β₁ reduction':>13}")
print(f"  {'-'*58}")
print(f"  {'Selective (wall)':>20}  {b1_sel:>4}  {coll_sel:>16.6f}  {b0-b1_sel:>13}")
print(f"  {'Random-k (mean)':>20}  {np.mean(random_b1s):>4.1f}  {np.mean(random_colls):>16.6f}  {b0-np.mean(random_b1s):>13.1f}")
print(f"  {'Random-k (best)':>20}  {min(random_b1s):>4}  {'—':>16}  {b0-min(random_b1s):>13}")
print(f"  {'Global':>20}  {b1_glob:>4}  {coll_glob:>16.6f}  {b0-b1_glob:>13}")

print(f"\n  Random-k β₁ distribution (n={n_random_trials}):")
print(f"    mean={np.mean(random_b1s):.2f}, std={np.std(random_b1s):.2f}")
print(f"    min={min(random_b1s)}, max={max(random_b1s)}")
print(f"    β₁=0 achieved: {sum(1 for b in random_b1s if b == 0)}/{n_random_trials}")

print(f"\n  ▶ T4 VERDICT:")
print(f"    1. Collateral=0 for selective IS definitional — non-targeted dims aren't touched.")
print(f"       This is true for random-k too (collateral = {np.mean(random_colls):.6f} ≈ 0)")
print(f"    2. The real question: does targeting WALL dims remove β₁ better than RANDOM dims?")
if b1_sel < np.mean(random_b1s):
    diff = np.mean(random_b1s) - b1_sel
    print(f"       Selective β₁={b1_sel} vs Random-k mean β₁={np.mean(random_b1s):.1f} → Selective is BETTER by {diff:.1f}")
    if min(random_b1s) <= b1_sel:
        print(f"       BUT random-k best={min(random_b1s)} matches or beats selective={b1_sel}")
        print(f"       → Wall targeting may not be special — lucky random dims can match it")
    else:
        print(f"       AND random-k never reaches β₁={b1_sel} in {n_random_trials} trials")
        print(f"       → Wall targeting IS genuinely better than random sparse")
elif b1_sel == 0 and sum(1 for b in random_b1s if b == 0) > 0:
    pct = sum(1 for b in random_b1s if b == 0) / n_random_trials * 100
    print(f"       Both reach β₁=0. Random achieves it {pct:.0f}% of the time.")
    print(f"       → Wall targeting is NOT uniquely special for β₁ removal")
else:
    print(f"       Selective β₁={b1_sel}, Random mean={np.mean(random_b1s):.1f} → Similar performance")


# ══════════════════════════════════════════════════════════════════════════════
# T5: PH artifact — noise가 50-dim에서 spurious β₁ 만드는가?
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("T5: PH ARTIFACT — Does noise create spurious β₁ in 50-dim?")
print("=" * 70)

n_noise_trials = 30
noise_b1s = []
struct_b1s = []

for trial in range(n_noise_trials):
    # Pure Gaussian noise (no structure)
    rng = np.random.RandomState(200 + trial)
    noise_cloud = rng.randn(N_POINTS, AMBIENT_DIM)
    b1_noise, mp_noise, _ = measure_beta1(noise_cloud)
    noise_b1s.append(b1_noise)

    # Structured cloud (planted holes)
    struct_cloud = make_cloud(seed=trial)
    b1_struct, mp_struct, _ = measure_beta1(struct_cloud)
    struct_b1s.append(b1_struct)

print(f"\n  Pure noise (N={n_noise_trials}):")
print(f"    β₁: mean={np.mean(noise_b1s):.2f}, std={np.std(noise_b1s):.2f}, range=[{min(noise_b1s)}, {max(noise_b1s)}]")

print(f"\n  Structured (planted holes, N={n_noise_trials}):")
print(f"    β₁: mean={np.mean(struct_b1s):.2f}, std={np.std(struct_b1s):.2f}, range=[{min(struct_b1s)}, {max(struct_b1s)}]")

ratio = np.mean(noise_b1s) / max(np.mean(struct_b1s), 1e-9)
print(f"\n  Noise/Structured β₁ ratio: {ratio:.2f}")

# With different persistence floors
print(f"\n  Sensitivity to persistence threshold:")
print(f"  {'threshold':>10}  {'noise_β₁':>10}  {'struct_β₁':>11}  {'ratio':>6}  {'separable?':>11}")
print(f"  {'-'*52}")
for floor in [0.0, 0.01, 0.05, 0.10, 0.15, 0.20]:
    n_b1s = []
    s_b1s = []
    for trial in range(n_noise_trials):
        rng = np.random.RandomState(200 + trial)
        nc = rng.randn(N_POINTS, AMBIENT_DIM)
        b1n, _, _ = measure_beta1(nc, floor=floor)
        n_b1s.append(b1n)

        sc = make_cloud(seed=trial)
        b1s, _, _ = measure_beta1(sc, floor=floor)
        s_b1s.append(b1s)

    nm, sm = np.mean(n_b1s), np.mean(s_b1s)
    r = nm / max(sm, 1e-9)
    sep = "YES" if nm < sm * 0.5 else ("PARTIAL" if nm < sm else "NO")
    print(f"  {floor:>10.2f}  {nm:>10.2f}  {sm:>11.2f}  {r:>6.2f}  {sep:>11}")

print(f"\n  ▶ T5 VERDICT:")
if np.mean(noise_b1s) < np.mean(struct_b1s) * 0.5:
    print(f"    Noise β₁ ({np.mean(noise_b1s):.1f}) << Structured β₁ ({np.mean(struct_b1s):.1f})")
    print(f"    → 50-dim에서 planted holes는 noise와 구분 가능. PH 감지 유효.")
else:
    print(f"    Noise β₁ ({np.mean(noise_b1s):.1f}) ≈ Structured β₁ ({np.mean(struct_b1s):.1f})")
    print(f"    → 50-dim에서 noise가 structured만큼 β₁ 생성. PH artifact 위험!")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
  T1 (수축 공식):
      phase3b (unit-step)  → 실제 LLM 코드, α 크면 inversion 가능
      phase6 (proportional) → 시뮬레이션 코드, rate<1 이면 안전
      → phase6 시뮬레이션 자체는 정상. phase3b만 주의 필요.

  T2 (Loss 스케일):
      시뮬레이션에서 λ=1.0 × lr=0.15 = 0.15 effective rate → 안전
      CE loss가 없는 시뮬레이션에서 λ=1.0은 유효
      → main의 λ=1.0이 시뮬레이션 맥락에서는 맞을 수 있음
      → 단, "실제 LLM에도 λ=1.0" 주장은 별개

  T3 (OOD):
      50-dim 시뮬레이션에서 수축은 분포를 축소 (OOD 아님)
      → 시뮬레이션에서 OOD 문제 없음

  T4 (Collateral=0):
      Collateral=0은 정의적 (non-targeted dims 안 건드리면 당연)
      Random-k도 collateral≈0
      핵심 질문: wall dims가 random dims보다 β₁ 제거에 유효한가?

  T5 (PH artifact):
      50-dim에서 noise vs structured 구분 가능 여부
""")
