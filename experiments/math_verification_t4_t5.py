"""
T4 + T5 완벽 수학 검증
======================

T4: Wall dims vs Random dims — 12스텝 전체 루프, 다중 시드, 통계 검정
T5: PH artifact — noise vs structured, persistence 분포, max persistence 비교

양쪽(Global, Selective) 모두 검증.
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy import stats
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

AMBIENT_DIM = 50
N_POINTS_RANGE = (30, 44)
N_STEPS = 12
RATE = 0.15
PERSISTENCE_FLOOR = 0.05

# Wall dims (프로젝트 동일)
RNG_MAP = np.random.RandomState(42)
WALL_DIMS = sorted(RNG_MAP.choice(AMBIENT_DIM, size=10, replace=False).tolist())
NON_WALL_DIMS = sorted(set(range(AMBIENT_DIM)) - set(WALL_DIMS))

print("=" * 70)
print("T4 + T5 COMPLETE MATH VERIFICATION")
print("=" * 70)
print(f"Config: {AMBIENT_DIM}-dim, {len(WALL_DIMS)} wall dims, {N_STEPS} steps, rate={RATE}")
print(f"Wall dims: {WALL_DIMS}")


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_cloud(n_pts, n_dim, rng, n_holes=2):
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


def measure(points, floor=PERSISTENCE_FLOOR):
    res = ripser(points, maxdim=1)
    dgm = res["dgms"][1]
    if len(dgm) == 0:
        return 0, 0.0, 0.0, []
    pers = dgm[:, 1] - dgm[:, 0]
    sig = pers[pers > floor]
    all_pers = sorted(pers.tolist(), reverse=True)
    return len(sig), float(sig.max()) if len(sig) > 0 else 0.0, float(sig.sum()), all_pers


def contract_selective(pts, dims, rate):
    """주어진 dims만 수축"""
    out = pts.copy()
    center = pts[:, dims].mean(axis=0)
    out[:, dims] = pts[:, dims] - rate * (pts[:, dims] - center)
    return out


def contract_global(pts, rate):
    center = pts.mean(axis=0)
    return pts - rate * (pts - center)


def contract_none(pts, rate):
    return pts.copy()


# ══════════════════════════════════════════════════════════════════════════════
# T4: WALL vs RANDOM — 12-step full loop
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("T4: WALL vs RANDOM SPARSE — 12-step full comparison")
print("=" * 70)

N_CLOUD_TRIALS = 8      # 프로젝트와 동일
N_RANDOM_SETS = 30       # random dim 조합 수

# 각 전략의 최종 β₁ 모으기
final_results = {
    "baseline": [],
    "global": [],
    "selective_wall": [],
    "selective_random_mean": [],
    "selective_random_best": [],
}

# 전체 trajectory 저장 (마지막에 평균 플롯용)
trajectories = {
    "baseline": [],
    "global": [],
    "selective_wall": [],
    "selective_random": [],  # 각 trial의 random mean
}

for trial in range(N_CLOUD_TRIALS):
    rng = np.random.RandomState(trial)
    n_pts = rng.randint(*N_POINTS_RANGE)
    cloud = make_cloud(n_pts, AMBIENT_DIM, rng, n_holes=2)

    b0, _, _, _ = measure(cloud)
    print(f"\n── Trial {trial} ({n_pts} pts, initial β₁={b0})")

    # === Baseline ===
    pts = cloud.copy()
    traj_base = [b0]
    for step in range(N_STEPS):
        pts = contract_none(pts, RATE)
        b, _, _, _ = measure(pts)
        traj_base.append(b)
    final_results["baseline"].append(traj_base[-1])
    trajectories["baseline"].append(traj_base)

    # === Global ===
    pts = cloud.copy()
    traj_glob = [b0]
    for step in range(N_STEPS):
        pts = contract_global(pts, RATE)
        b, _, _, _ = measure(pts)
        traj_glob.append(b)
    final_results["global"].append(traj_glob[-1])
    trajectories["global"].append(traj_glob)
    print(f"  Global:     β₁ {b0} → {traj_glob[-1]}")

    # === Selective (wall dims) ===
    pts = cloud.copy()
    traj_wall = [b0]
    for step in range(N_STEPS):
        pts = contract_selective(pts, WALL_DIMS, RATE)
        b, _, _, _ = measure(pts)
        traj_wall.append(b)
    final_results["selective_wall"].append(traj_wall[-1])
    trajectories["selective_wall"].append(traj_wall)
    print(f"  Wall:       β₁ {b0} → {traj_wall[-1]}")

    # === Selective (random k dims) — 여러 조합 ===
    random_finals = []
    random_trajs = []
    for r_trial in range(N_RANDOM_SETS):
        r_rng = np.random.RandomState(1000 + trial * 100 + r_trial)
        random_dims = sorted(r_rng.choice(AMBIENT_DIM, size=len(WALL_DIMS), replace=False).tolist())

        pts = cloud.copy()
        traj_rand = [b0]
        for step in range(N_STEPS):
            pts = contract_selective(pts, random_dims, RATE)
            b, _, _, _ = measure(pts)
            traj_rand.append(b)
        random_finals.append(traj_rand[-1])
        random_trajs.append(traj_rand)

    mean_random_final = np.mean(random_finals)
    best_random_final = min(random_finals)
    final_results["selective_random_mean"].append(mean_random_final)
    final_results["selective_random_best"].append(best_random_final)

    # random의 trial별 mean trajectory
    mean_traj = np.mean(random_trajs, axis=0).tolist()
    trajectories["selective_random"].append(mean_traj)

    pct_zero = sum(1 for f in random_finals if f == 0) / len(random_finals) * 100
    print(f"  Random-k:   β₁ {b0} → mean {mean_random_final:.1f}, best {best_random_final}, "
          f"β₁=0: {pct_zero:.0f}% ({sum(1 for f in random_finals if f==0)}/{N_RANDOM_SETS})")

# ── T4 통계 요약 ────────────────────────────────────────────────────────────

print("\n\n" + "=" * 70)
print("T4 RESULTS SUMMARY (12 steps × 8 trials)")
print("=" * 70)

print(f"\n  {'Strategy':>25}  {'final β₁ mean':>14}  {'std':>6}  {'min':>4}  {'max':>4}  {'β₁=0 rate':>10}")
print(f"  {'-'*70}")

for name in ["baseline", "global", "selective_wall", "selective_random_mean", "selective_random_best"]:
    vals = final_results[name]
    z_rate = sum(1 for v in vals if v == 0) / len(vals) * 100
    print(f"  {name:>25}  {np.mean(vals):>14.2f}  {np.std(vals):>6.2f}  {min(vals):>4.0f}  {max(vals):>4.0f}  {z_rate:>9.0f}%")

# Wall vs Random 통계 검정
wall_finals = np.array(final_results["selective_wall"], dtype=float)
rand_finals = np.array(final_results["selective_random_mean"], dtype=float)

if np.std(wall_finals) > 0 or np.std(rand_finals) > 0:
    t_stat, p_value = stats.ttest_ind(wall_finals, rand_finals)
    # Mann-Whitney (비모수)
    u_stat, p_mann = stats.mannwhitneyu(wall_finals, rand_finals, alternative='less')
    print(f"\n  Wall vs Random-k (mean):")
    print(f"    t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"    Mann-Whitney (wall < random): U={u_stat:.1f}, p={p_mann:.4f}")
else:
    print(f"\n  Wall vs Random-k: 분산 0 — 동일 결과")

# Step-by-step trajectory 비교
print(f"\n  Step-by-step β₁ trajectory (mean across trials):")
print(f"  {'Step':>5}  {'Baseline':>9}  {'Global':>8}  {'Wall':>6}  {'Random-k':>9}  {'Wall-Rand':>10}")
print(f"  {'-'*50}")
for step in range(N_STEPS + 1):
    b_base = np.mean([t[step] for t in trajectories["baseline"]])
    b_glob = np.mean([t[step] for t in trajectories["global"]])
    b_wall = np.mean([t[step] for t in trajectories["selective_wall"]])
    b_rand = np.mean([t[step] for t in trajectories["selective_random"]])
    diff = b_wall - b_rand
    print(f"  {step:>5}  {b_base:>9.2f}  {b_glob:>8.2f}  {b_wall:>6.2f}  {b_rand:>9.2f}  {diff:>+10.2f}")

print(f"\n  ▶ T4 VERDICT:")
w_mean = np.mean(wall_finals)
r_mean = np.mean(rand_finals)
if w_mean < r_mean - 0.5:
    print(f"    Wall ({w_mean:.2f}) < Random ({r_mean:.2f}) → Wall targeting IS better")
elif abs(w_mean - r_mean) <= 0.5:
    print(f"    Wall ({w_mean:.2f}) ≈ Random ({r_mean:.2f}) → Wall targeting NOT special")
    print(f"    'Selective가 Global보다 좋다'는 sparse intervention 효과이지,")
    print(f"    wall neuron을 정확히 겨냥한 효과가 아닐 수 있음")
else:
    print(f"    Wall ({w_mean:.2f}) > Random ({r_mean:.2f}) → Wall targeting WORSE than random!")


# ══════════════════════════════════════════════════════════════════════════════
# T5: PH ARTIFACT — 완벽 검증
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("T5: PH ARTIFACT — COMPLETE VERIFICATION")
print("=" * 70)

N_ARTIFACT_TRIALS = 50

# ── 5a: β₁ count 비교 ──────────────────────────────────────────────────────

print("\n── 5a: β₁ COUNT comparison (N=50 trials)")

noise_b1s = []
struct_b1s = []
noise_max_pers = []
struct_max_pers = []
noise_total_pers = []
struct_total_pers = []
noise_all_pers = []
struct_all_pers = []

for trial in range(N_ARTIFACT_TRIALS):
    rng_n = np.random.RandomState(500 + trial)
    n_pts = rng_n.randint(*N_POINTS_RANGE)

    # Pure noise
    noise_cloud = rng_n.randn(n_pts, AMBIENT_DIM)
    b1n, mpn, tpn, all_pn = measure(noise_cloud)
    noise_b1s.append(b1n)
    noise_max_pers.append(mpn)
    noise_total_pers.append(tpn)
    noise_all_pers.extend(all_pn[:5])  # top-5 persistences

    # Structured (planted holes)
    struct_cloud = make_cloud(n_pts, AMBIENT_DIM, rng_n, n_holes=2)
    b1s, mps, tps, all_ps = measure(struct_cloud)
    struct_b1s.append(b1s)
    struct_max_pers.append(mps)
    struct_total_pers.append(tps)
    struct_all_pers.extend(all_ps[:5])

print(f"\n  {'':>20}  {'Noise':>12}  {'Structured':>12}  {'Noise/Struct':>13}")
print(f"  {'-'*60}")
print(f"  {'β₁ count mean':>20}  {np.mean(noise_b1s):>12.2f}  {np.mean(struct_b1s):>12.2f}  {np.mean(noise_b1s)/max(np.mean(struct_b1s),1e-9):>13.2f}x")
print(f"  {'β₁ count std':>20}  {np.std(noise_b1s):>12.2f}  {np.std(struct_b1s):>12.2f}")
print(f"  {'β₁ count range':>20}  [{min(noise_b1s)}-{max(noise_b1s)}]{'':>4}  [{min(struct_b1s)}-{max(struct_b1s)}]")
print(f"  {'max pers mean':>20}  {np.mean(noise_max_pers):>12.4f}  {np.mean(struct_max_pers):>12.4f}  {np.mean(noise_max_pers)/max(np.mean(struct_max_pers),1e-9):>13.2f}x")
print(f"  {'total pers mean':>20}  {np.mean(noise_total_pers):>12.4f}  {np.mean(struct_total_pers):>12.4f}  {np.mean(noise_total_pers)/max(np.mean(struct_total_pers),1e-9):>13.2f}x")

# ── 5b: Persistence 분포 비교 ──────────────────────────────────────────────

print(f"\n── 5b: PERSISTENCE DISTRIBUTION")
print(f"\n  Top-5 persistence values distribution:")
print(f"  {'':>15}  {'Noise':>12}  {'Structured':>12}")
print(f"  {'-'*42}")

noise_pers_arr = np.array(noise_all_pers)
struct_pers_arr = np.array(struct_all_pers)

for pct in [25, 50, 75, 90, 95, 99]:
    np_val = np.percentile(noise_pers_arr, pct) if len(noise_pers_arr) > 0 else 0
    sp_val = np.percentile(struct_pers_arr, pct) if len(struct_pers_arr) > 0 else 0
    print(f"  {'p'+str(pct):>15}  {np_val:>12.4f}  {sp_val:>12.4f}")

# KS test
if len(noise_pers_arr) > 0 and len(struct_pers_arr) > 0:
    ks_stat, ks_p = stats.ks_2samp(noise_pers_arr, struct_pers_arr)
    print(f"\n  KS test (noise vs struct persistence): stat={ks_stat:.4f}, p={ks_p:.6f}")

# ── 5c: MAX persistence가 핵심 ─────────────────────────────────────────────

print(f"\n── 5c: MAX PERSISTENCE — the key discriminator")
print(f"\n  이 프로젝트가 감지하는 'wall'은 max persistence가 높은 β₁ bar.")
print(f"  noise의 max persistence vs structured의 max persistence 비교:")

print(f"\n  {'':>20}  {'Noise':>12}  {'Structured':>12}")
print(f"  {'-'*48}")
print(f"  {'max pers mean':>20}  {np.mean(noise_max_pers):>12.4f}  {np.mean(struct_max_pers):>12.4f}")
print(f"  {'max pers std':>20}  {np.std(noise_max_pers):>12.4f}  {np.std(struct_max_pers):>12.4f}")
print(f"  {'max pers p50':>20}  {np.median(noise_max_pers):>12.4f}  {np.median(struct_max_pers):>12.4f}")
print(f"  {'max pers p95':>20}  {np.percentile(noise_max_pers,95):>12.4f}  {np.percentile(struct_max_pers,95):>12.4f}")

# t-test on max persistence
t_mp, p_mp = stats.ttest_ind(noise_max_pers, struct_max_pers)
u_mp, p_ump = stats.mannwhitneyu(struct_max_pers, noise_max_pers, alternative='greater')
print(f"\n  Max persistence comparison:")
print(f"    t-test: t={t_mp:.3f}, p={p_mp:.6f}")
print(f"    Mann-Whitney (struct > noise): U={u_mp:.1f}, p={p_ump:.6f}")

# ── 5d: Persistence threshold로 분리 가능? ──────────────────────────────────

print(f"\n── 5d: CAN PERSISTENCE THRESHOLD SEPARATE NOISE FROM STRUCTURE?")
print(f"\n  threshold에 따른 β₁ count:")
print(f"  {'threshold':>10}  {'noise_β₁':>10}  {'struct_β₁':>11}  {'noise>struct':>13}  {'separable':>10}")
print(f"  {'-'*58}")

for floor in [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00]:
    n_vals = []
    s_vals = []
    for trial in range(N_ARTIFACT_TRIALS):
        rng_n = np.random.RandomState(500 + trial)
        n_pts = rng_n.randint(*N_POINTS_RANGE)

        noise_cloud = rng_n.randn(n_pts, AMBIENT_DIM)
        b1n, _, _, _ = measure(noise_cloud, floor=floor)
        n_vals.append(b1n)

        struct_cloud = make_cloud(n_pts, AMBIENT_DIM, rng_n, n_holes=2)
        b1s, _, _, _ = measure(struct_cloud, floor=floor)
        s_vals.append(b1s)

    nm, sm = np.mean(n_vals), np.mean(s_vals)
    bigger = "YES" if nm > sm else "NO"
    if nm == 0 and sm == 0:
        sep = "BOTH=0"
    elif nm < sm * 0.3:
        sep = "YES"
    elif nm < sm:
        sep = "PARTIAL"
    else:
        sep = "NO"
    print(f"  {floor:>10.2f}  {nm:>10.2f}  {sm:>11.2f}  {bigger:>13}  {sep:>10}")

# ── 5e: 차원 효과 ──────────────────────────────────────────────────────────

print(f"\n── 5e: DIMENSION EFFECT — β₁ vs ambient dimension")
print(f"\n  고차원에서 noise β₁이 더 심각해지는가?")
print(f"  {'dim':>5}  {'noise_β₁':>10}  {'struct_β₁':>11}  {'ratio':>7}")
print(f"  {'-'*36}")

for dim in [10, 20, 30, 50, 100, 200]:
    n_vals = []
    s_vals = []
    for trial in range(20):
        rng_d = np.random.RandomState(800 + trial)
        n_pts = rng_d.randint(30, 44)

        # noise
        nc = rng_d.randn(n_pts, dim)
        b1n, _, _, _ = measure(nc)
        n_vals.append(b1n)

        # structured (holes in first 10 dims)
        sc = rng_d.randn(n_pts, dim) * 0.05
        wall_d = list(range(min(10, dim)))
        pts_per_hole = n_pts // 2
        for h in range(2):
            if len(wall_d) >= 2:
                pd = rng_d.choice(wall_d, size=2, replace=False)
                i0, i1 = h * pts_per_hole, (h+1) * pts_per_hole
                theta = np.linspace(0, 2*np.pi, i1-i0, endpoint=False)
                radius = 1.0 + 0.1*rng_d.randn(len(theta))
                sc[i0:i1, pd[0]] = radius * np.cos(theta)
                sc[i0:i1, pd[1]] = radius * np.sin(theta)
        b1s, _, _, _ = measure(sc)
        s_vals.append(b1s)

    nm, sm = np.mean(n_vals), np.mean(s_vals)
    ratio = nm / max(sm, 1e-9)
    print(f"  {dim:>5}  {nm:>10.2f}  {sm:>11.2f}  {ratio:>7.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │ T4: WALL vs RANDOM SPARSE CONTROL                               │
  ├──────────────────────────────────────────────────────────────────┤
  │ Wall final β₁ (mean): {np.mean(final_results['selective_wall']):>6.2f}                                  │
  │ Random final β₁ (mean): {np.mean(final_results['selective_random_mean']):>6.2f}                                │
  │ Difference: {np.mean(final_results['selective_wall']) - np.mean(final_results['selective_random_mean']):>+6.2f}                                           │
  │                                                                  │""")

if abs(np.mean(final_results['selective_wall']) - np.mean(final_results['selective_random_mean'])) < 0.5:
    print(f"  │ VERDICT: Wall targeting ≈ Random targeting                      │")
    print(f"  │ → 'Selective가 좋다'는 sparse intervention 효과                  │")
    print(f"  │ → Wall neuron 특이성은 증명 안 됨                                │")
else:
    w = np.mean(final_results['selective_wall'])
    r = np.mean(final_results['selective_random_mean'])
    if w < r:
        print(f"  │ VERDICT: Wall targeting IS better than random                    │")
    else:
        print(f"  │ VERDICT: Wall targeting WORSE than random                        │")

print(f"""  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ T5: PH ARTIFACT IN {AMBIENT_DIM}-DIM                                         │
  ├──────────────────────────────────────────────────────────────────┤
  │ Noise β₁ (mean): {np.mean(noise_b1s):>6.2f}                                      │
  │ Structured β₁ (mean): {np.mean(struct_b1s):>6.2f}                                  │
  │ Noise max persistence: {np.mean(noise_max_pers):>.4f}                                │
  │ Struct max persistence: {np.mean(struct_max_pers):>.4f}                               │
  │                                                                  │""")

if np.mean(noise_b1s) > np.mean(struct_b1s):
    print(f"  │ β₁ COUNT: Noise > Structured → β₁ count만으로 구조 판별 불가     │")
else:
    print(f"  │ β₁ COUNT: Structured > Noise → β₁ count로 구조 판별 가능         │")

if np.mean(struct_max_pers) > np.mean(noise_max_pers) * 1.5:
    print(f"  │ MAX PERSISTENCE: Structured >> Noise → persistence로 구분 가능!  │")
    print(f"  │ → β₁ count가 아니라 max persistence가 진짜 신호                  │")
elif np.mean(struct_max_pers) > np.mean(noise_max_pers):
    print(f"  │ MAX PERSISTENCE: Structured > Noise → 약한 구분 가능             │")
else:
    print(f"  │ MAX PERSISTENCE: Noise ≥ Structured → persistence도 구분 불가!   │")
    print(f"  │ → PH 감지 전체가 artifact 위험                                  │")

print(f"  └──────────────────────────────────────────────────────────────────┘")

# Impact on project claims
print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │ IMPACT ON PROJECT CLAIMS                                         │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │ 1. "Selective가 β₁=0 달성"                                      │
  │    → T4가 random과 동일이면: sparse intervention 효과            │
  │    → T4가 wall이 더 좋으면: wall targeting 효과 (claim 유지)     │
  │                                                                  │
  │ 2. "β₁ hole이 벽 역할"                                          │
  │    → T5에서 noise>structured이면: 감지된 β₁이 noise일 수 있음   │
  │    → max persistence로 구분되면: 높은 pers만 진짜 벽으로 해석    │
  │                                                                  │
  │ 3. "Collateral damage = 0"                                       │
  │    → T4에서 random-k도 collateral=0이면: 정의적 (claim 무의미)   │
  │                                                                  │
  │ Global에도 동일 적용:                                             │
  │    Global의 β₁ 감소도 noise β₁ 감소일 수 있음 (T5)              │
  │    Global의 collateral은 실제이지만, 줄인 β₁의 의미가 불확실     │
  └──────────────────────────────────────────────────────────────────┘
""")
