"""
Phase 6 — Three-Way Comparison: Baseline vs Global vs Selective
================================================================

Compares three strategies on identical point clouds:
  1. Baseline  — no contraction at all (original model)
  2. Global    — radial contraction on ALL dimensions
  3. Selective — radial contraction on ONLY wall-neuron dimensions

Metrics per step:
  - β₁ count (wall count)
  - Max persistence (wall strength)
  - Total persistence (cumulative wall energy)
  - Collateral damage (non-wall dim distortion)
  - Wall-dim signal (mean pairwise distance in wall dims — should decrease)
  - Non-wall-dim signal (mean pairwise distance in non-wall dims — should stay constant)
"""

import sys, pathlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.spatial.distance import pdist

# ── Configuration ────────────────────────────────────────────────────────────

AMBIENT_DIM = 50
N_POINTS_RANGE = (30, 44)
N_TRIALS = 8
N_STEPS = 12
CONTRACTION_RATE = 0.15
PERSISTENCE_FLOOR = 0.05

CREATIVE_DIMS = [406, 3884, 3433, 940, 3951]
REASONING_DIMS = [1917, 2720, 2977, 866, 133]
ALL_WALL_DIMS_FULL = sorted(set(CREATIVE_DIMS + REASONING_DIMS))

RNG_MAP = np.random.RandomState(42)
WALL_DIMS = sorted(RNG_MAP.choice(AMBIENT_DIM, size=len(ALL_WALL_DIMS_FULL), replace=False).tolist())
NON_WALL_DIMS = sorted(set(range(AMBIENT_DIM)) - set(WALL_DIMS))

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

print(f"Wall dims ({len(WALL_DIMS)}): {WALL_DIMS}")
print(f"Non-wall dims: {len(NON_WALL_DIMS)}")


# ── Point cloud generation ───────────────────────────────────────────────────

def make_cloud_with_holes(n_points, n_dim, rng, n_holes=2):
    points = rng.randn(n_points, n_dim) * 0.05
    pts_per_hole = n_points // n_holes
    for h in range(n_holes):
        plane_dims = rng.choice(WALL_DIMS, size=2, replace=False)
        i0 = h * pts_per_hole
        i1 = i0 + pts_per_hole
        theta = np.linspace(0, 2 * np.pi, i1 - i0, endpoint=False)
        radius = 1.0 + 0.1 * rng.randn(len(theta))
        points[i0:i1, plane_dims[0]] = radius * np.cos(theta)
        points[i0:i1, plane_dims[1]] = radius * np.sin(theta)
    return points


# ── Measurement ──────────────────────────────────────────────────────────────

def measure_topology(points):
    res = ripser(points, maxdim=1)
    dgm = res["dgms"][1]
    if len(dgm) == 0:
        return 0, 0.0, 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    sig = pers[pers > PERSISTENCE_FLOOR]
    beta1 = len(sig)
    max_pers = float(sig.max()) if beta1 > 0 else 0.0
    total_pers = float(sig.sum()) if beta1 > 0 else 0.0
    return beta1, max_pers, total_pers


def collateral_damage(original, modified):
    d_orig = pdist(original[:, NON_WALL_DIMS])
    d_mod = pdist(modified[:, NON_WALL_DIMS])
    return float(np.mean(np.abs(d_orig - d_mod)))


def wall_dim_signal(points):
    return float(np.mean(pdist(points[:, WALL_DIMS])))


def non_wall_dim_signal(points):
    return float(np.mean(pdist(points[:, NON_WALL_DIMS])))


# ── Contraction strategies ───────────────────────────────────────────────────

def contract_none(points, rate):
    """Baseline: no contraction."""
    return points.copy()


def contract_global(points, rate):
    """Global: radial contraction on ALL dimensions."""
    center = points.mean(axis=0)
    return points - rate * (points - center)


def contract_selective(points, rate):
    """Selective: radial contraction on ONLY wall-neuron dimensions."""
    out = points.copy()
    center = points[:, WALL_DIMS].mean(axis=0)
    out[:, WALL_DIMS] = points[:, WALL_DIMS] - rate * (points[:, WALL_DIMS] - center)
    return out


STRATEGIES = [
    ("baseline",  contract_none,      "#1f77b4", "^"),
    ("global",    contract_global,    "#d62728", "o"),
    ("selective", contract_selective, "#2ca02c", "s"),
]


# ── Experiment ───────────────────────────────────────────────────────────────

def run_experiment():
    results = {
        name: {"beta1": [], "max_pers": [], "total_pers": [],
               "collateral": [], "wall_signal": [], "nonwall_signal": []}
        for name, _, _, _ in STRATEGIES
    }

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial)
        n_pts = rng.randint(*N_POINTS_RANGE)
        cloud = make_cloud_with_holes(n_pts, AMBIENT_DIM, rng, n_holes=2)
        original = cloud.copy()

        b0, p0, tp0 = measure_topology(cloud)
        ws0 = wall_dim_signal(cloud)
        nws0 = non_wall_dim_signal(cloud)
        print(f"\n── Trial {trial} ({n_pts} pts)  β₁={b0}  max_pers={p0:.3f}  total_pers={tp0:.3f}")

        for name, fn, _, _ in STRATEGIES:
            pts = cloud.copy()
            rec = results[name]
            rec["beta1"].append([b0])
            rec["max_pers"].append([p0])
            rec["total_pers"].append([tp0])
            rec["collateral"].append([0.0])
            rec["wall_signal"].append([ws0])
            rec["nonwall_signal"].append([nws0])

            for step in range(1, N_STEPS + 1):
                pts = fn(pts, CONTRACTION_RATE)
                b, p, tp = measure_topology(pts)
                c = collateral_damage(original, pts)
                ws = wall_dim_signal(pts)
                nws = non_wall_dim_signal(pts)
                rec["beta1"][-1].append(b)
                rec["max_pers"][-1].append(p)
                rec["total_pers"][-1].append(tp)
                rec["collateral"][-1].append(c)
                rec["wall_signal"][-1].append(ws)
                rec["nonwall_signal"][-1].append(nws)

            final = rec["beta1"][-1][-1]
            coll = rec["collateral"][-1][-1]
            print(f"  {name:10s}  β₁={final}  collateral={coll:.4f}")

    return results


def plot_results(results):
    steps = np.arange(N_STEPS + 1)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    metrics = [
        ("beta1",        "β₁ Count (벽 개수)",            axes[0, 0]),
        ("max_pers",     "Max Persistence (벽 강도)",      axes[0, 1]),
        ("total_pers",   "Total Persistence (벽 에너지)",   axes[0, 2]),
        ("collateral",   "Collateral Damage\n(non-wall Δ)", axes[1, 0]),
        ("wall_signal",  "Wall-Dim Signal\n(wall 평균 거리)", axes[1, 1]),
        ("nonwall_signal", "Non-Wall-Dim Signal\n(non-wall 평균 거리)", axes[1, 2]),
    ]

    for key, ylabel, ax in metrics:
        for name, _, color, marker in STRATEGIES:
            arr = np.array(results[name][key], dtype=float)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(steps, mean, color=color, marker=marker, markersize=4,
                    label=name, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)
        ax.set_xlabel("Contraction Step")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Baseline vs Global vs Selective  —  Three-Way Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = DATA_DIR / "three_way_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    return out


def save_summary(results):
    summary = {}
    for name, _, _, _ in STRATEGIES:
        r = results[name]
        final_beta = np.array(r["beta1"])[:, -1]
        final_coll = np.array(r["collateral"])[:, -1]
        final_maxp = np.array(r["max_pers"])[:, -1]
        final_totp = np.array(r["total_pers"])[:, -1]
        final_ws = np.array(r["wall_signal"])[:, -1]
        final_nws = np.array(r["nonwall_signal"])[:, -1]
        init_beta = np.array(r["beta1"])[:, 0]
        init_ws = np.array(r["wall_signal"])[:, 0]
        init_nws = np.array(r["nonwall_signal"])[:, 0]

        summary[name] = {
            "initial_beta1": f"{init_beta.mean():.2f} ± {init_beta.std():.2f}",
            "final_beta1": f"{final_beta.mean():.2f} ± {final_beta.std():.2f}",
            "beta1_reduction": f"{((init_beta.mean() - final_beta.mean()) / max(init_beta.mean(), 1e-9)) * 100:.1f}%",
            "final_max_persistence": f"{final_maxp.mean():.4f} ± {final_maxp.std():.4f}",
            "final_total_persistence": f"{final_totp.mean():.4f} ± {final_totp.std():.4f}",
            "collateral_damage": f"{final_coll.mean():.4f} ± {final_coll.std():.4f}",
            "wall_signal_initial": f"{init_ws.mean():.4f}",
            "wall_signal_final": f"{final_ws.mean():.4f}",
            "wall_signal_reduction": f"{((init_ws.mean() - final_ws.mean()) / max(init_ws.mean(), 1e-9)) * 100:.1f}%",
            "nonwall_signal_initial": f"{init_nws.mean():.4f}",
            "nonwall_signal_final": f"{final_nws.mean():.4f}",
            "nonwall_signal_change": f"{((final_nws.mean() - init_nws.mean()) / max(init_nws.mean(), 1e-9)) * 100:.1f}%",
        }

    out = DATA_DIR / "three_way_comparison.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON saved → {out}")
    return summary


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)
    summary = save_summary(results)

    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'':12s} {'Baseline':>20s} {'Global':>20s} {'Selective':>20s}")
    print("-" * 70)
    for key in ["final_beta1", "collateral_damage", "beta1_reduction",
                "wall_signal_reduction", "nonwall_signal_change"]:
        label = key.replace("_", " ").title()
        vals = [summary[n][key] for n in ["baseline", "global", "selective"]]
        print(f"  {label:22s} {vals[0]:>16s} {vals[1]:>16s} {vals[2]:>16s}")
    print()
