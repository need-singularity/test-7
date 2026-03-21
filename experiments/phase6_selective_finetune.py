"""
Phase 6 — Selective Neuron Fine-Tuning Simulation
==================================================

Hypothesis: fine-tuning ONLY wall-neuron dimensions (selective / targeted LoRA)
collapses topological walls faster and with less collateral damage than
perturbing all dimensions equally (global fine-tuning).

Wall-neuron groups (from Phase 5 analysis):
  Creative/boundary : dims 406, 3884, 3433, 940, 3951
  Reasoning/factual : dims 1917, 2720, 2977, 866, 133

Methodology:
  1. Synthesise point clouds in 50-dim with planted β₁ holes.
  2. Two strategies — global vs selective radial contraction.
  3. At each step measure β₁ count, max persistence, and collateral damage
     (mean pairwise-distance change in non-wall dimensions).
  4. Compare across multiple random seeds and plot results.
"""

import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ripser import ripser

# ── Configuration ────────────────────────────────────────────────────────────

AMBIENT_DIM = 50                  # matches Phase 1b
N_POINTS_RANGE = (30, 44)         # same range as Phase 1b
N_TRIALS = 8                      # random seeds
N_STEPS = 12                      # contraction iterations
CONTRACTION_RATE = 0.15           # fraction of radial distance removed per step
PERSISTENCE_FLOOR = 0.05          # ignore noise features below this

# Wall-neuron dimensions (mapped into 0..49 for our 50-dim simulation)
CREATIVE_DIMS = [406, 3884, 3433, 940, 3951]
REASONING_DIMS = [1917, 2720, 2977, 866, 133]
ALL_WALL_DIMS_FULL = sorted(set(CREATIVE_DIMS + REASONING_DIMS))

# Map to our 50-dim space: pick 10 unique dims deterministically
RNG_MAP = np.random.RandomState(42)
WALL_DIMS = sorted(RNG_MAP.choice(AMBIENT_DIM, size=len(ALL_WALL_DIMS_FULL), replace=False).tolist())
NON_WALL_DIMS = sorted(set(range(AMBIENT_DIM)) - set(WALL_DIMS))

print(f"Simulated wall dims   : {WALL_DIMS}")
print(f"Non-wall dims (count) : {len(NON_WALL_DIMS)}")


# ── Synthetic point-cloud generation ─────────────────────────────────────────

def make_cloud_with_holes(n_points: int, n_dim: int, rng: np.random.RandomState,
                          n_holes: int = 2) -> np.ndarray:
    """
    Create a point cloud in R^n_dim with planted β₁ holes.

    Strategy: place points on circles (in randomly chosen 2-d subspaces
    drawn from the WALL dims) so the holes are aligned with wall neurons,
    then add moderate Gaussian noise.
    """
    points = rng.randn(n_points, n_dim) * 0.05          # low-amplitude background

    pts_per_hole = n_points // n_holes
    for h in range(n_holes):
        # choose a 2-d plane inside the wall dims for each hole
        plane_dims = rng.choice(WALL_DIMS, size=2, replace=False)
        idx_start = h * pts_per_hole
        idx_end = idx_start + pts_per_hole
        theta = np.linspace(0, 2 * np.pi, idx_end - idx_start, endpoint=False)
        radius = 1.0 + 0.1 * rng.randn(len(theta))
        points[idx_start:idx_end, plane_dims[0]] = radius * np.cos(theta)
        points[idx_start:idx_end, plane_dims[1]] = radius * np.sin(theta)

    return points


# ── PH measurement helpers ───────────────────────────────────────────────────

def measure_topology(points: np.ndarray):
    """Return (β₁ count, max persistence) for significant features."""
    res = ripser(points, maxdim=1)
    dgm = res["dgms"][1]
    if len(dgm) == 0:
        return 0, 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    sig = pers[pers > PERSISTENCE_FLOOR]
    beta1 = len(sig)
    max_pers = float(sig.max()) if beta1 > 0 else 0.0
    return beta1, max_pers


def collateral_damage(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Mean absolute change in pairwise distances computed ONLY over non-wall dims.
    Measures how much the representation changes in dimensions we did not target.
    """
    orig_nw = original[:, NON_WALL_DIMS]
    mod_nw = modified[:, NON_WALL_DIMS]
    from scipy.spatial.distance import pdist
    d_orig = pdist(orig_nw)
    d_mod = pdist(mod_nw)
    return float(np.mean(np.abs(d_orig - d_mod)))


# ── Contraction strategies ───────────────────────────────────────────────────

def contract_global(points: np.ndarray, rate: float) -> np.ndarray:
    """Radial contraction applied equally to ALL dimensions."""
    center = points.mean(axis=0)
    directions = points - center
    return points - rate * directions            # pull every dim toward centroid


def contract_selective(points: np.ndarray, rate: float) -> np.ndarray:
    """Radial contraction applied ONLY to wall-neuron dimensions."""
    out = points.copy()
    center = points[:, WALL_DIMS].mean(axis=0)
    directions = points[:, WALL_DIMS] - center
    out[:, WALL_DIMS] = points[:, WALL_DIMS] - rate * directions
    return out


# ── Main experiment loop ─────────────────────────────────────────────────────

def run_experiment():
    results = {
        "global":    {"beta1": [], "max_pers": [], "collateral": []},
        "selective": {"beta1": [], "max_pers": [], "collateral": []},
    }

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial)
        n_pts = rng.randint(*N_POINTS_RANGE)
        cloud = make_cloud_with_holes(n_pts, AMBIENT_DIM, rng, n_holes=2)
        original_cloud = cloud.copy()

        b0, p0 = measure_topology(cloud)
        print(f"\n── Trial {trial} ({n_pts} pts)  initial β₁={b0}  max_pers={p0:.3f}")

        for label, contract_fn in [("global", contract_global),
                                   ("selective", contract_selective)]:
            pts = cloud.copy()
            betas, perss, colls = [b0], [p0], [0.0]
            for step in range(1, N_STEPS + 1):
                pts = contract_fn(pts, CONTRACTION_RATE)
                b, p = measure_topology(pts)
                c = collateral_damage(original_cloud, pts)
                betas.append(b)
                perss.append(p)
                colls.append(c)
            results[label]["beta1"].append(betas)
            results[label]["max_pers"].append(perss)
            results[label]["collateral"].append(colls)
            print(f"  {label:10s}  final β₁={betas[-1]}  max_pers={perss[-1]:.3f}  collateral={colls[-1]:.4f}")

    return results


def plot_results(results: dict, out_path: pathlib.Path):
    steps = np.arange(N_STEPS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for label, color, marker in [("global", "#d62728", "o"),
                                  ("selective", "#2ca02c", "s")]:
        beta_arr = np.array(results[label]["beta1"], dtype=float)
        pers_arr = np.array(results[label]["max_pers"], dtype=float)
        coll_arr = np.array(results[label]["collateral"], dtype=float)

        def _plot(ax, arr, ylabel):
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(steps, mean, color=color, marker=marker, markersize=4,
                    label=label, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_xlabel("Contraction step")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        _plot(axes[0], beta_arr, "β₁ count")
        _plot(axes[1], pers_arr, "Max persistence")
        _plot(axes[2], coll_arr, "Collateral damage\n(non-wall dim Δ dist)")

    axes[0].set_title("Wall Removal: β₁ Count")
    axes[1].set_title("Wall Removal: Max Persistence")
    axes[2].set_title("Collateral Damage")

    fig.suptitle("Selective vs Global Fine-Tuning  —  Topological Wall Collapse",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_experiment()
    out = pathlib.Path("/Users/ghost/Dev/hamster-ball/data/selective_vs_global.png")
    plot_results(results, out)

    # Summary statistics
    g_final = np.array(results["global"]["beta1"])[:, -1]
    s_final = np.array(results["selective"]["beta1"])[:, -1]
    g_coll = np.array(results["global"]["collateral"])[:, -1]
    s_coll = np.array(results["selective"]["collateral"])[:, -1]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Final β₁    global {g_final.mean():.2f} ± {g_final.std():.2f}   "
          f"selective {s_final.mean():.2f} ± {s_final.std():.2f}")
    print(f"  Collateral  global {g_coll.mean():.4f} ± {g_coll.std():.4f}   "
          f"selective {s_coll.mean():.4f} ± {s_coll.std():.4f}")

    if s_coll.mean() < g_coll.mean():
        ratio = g_coll.mean() / max(s_coll.mean(), 1e-9)
        print(f"\n  → Selective fine-tuning has {ratio:.1f}x LESS collateral damage")
    if s_final.mean() <= g_final.mean():
        print(f"  → Selective fine-tuning removes walls equally or better")
    else:
        print(f"  → Global fine-tuning removes more walls (but at higher cost)")
    print()
