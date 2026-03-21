"""
Phase 6: Wall Removal Fine-Tuning PoC — Topology Regularization Simulation

Inspired by Perelman's proof of the Poincare conjecture: instead of collapsing
walls after they form, PREVENT them from forming by adding a topology
regularization term to the training loss:

    L_total = L_task + lambda * L_topology
    L_topology = sum of persistence(beta_1 bars)

Since real LoRA fine-tuning requires GPU/PyTorch, this PoC simulates the
gradient descent process on synthetic point clouds derived from real LLM
topology data.  We:

  1. Load passage directions + wall centers from Phase 2
  2. Build synthetic point clouds that replicate the beta_1 structure
     observed in Llama-3.1-8B embeddings
  3. Iteratively apply radial contraction toward cycle centroids
     (simulating the gradient of L_topology)
  4. Track beta_1 count, max persistence, and beta_0 stability per step
  5. Compare convergence across lambda values (0.01, 0.1, 0.5, 1.0)
  6. Generate a convergence plot

Usage:
    cd /Users/ghost/Dev/hamster-ball
    source .venv/bin/activate
    python -m experiments.phase6_wall_finetune
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/Users/ghost/Dev/test-7/data/poc_topology")
PASSAGE_DIR_PATH = DATA_ROOT / "passage_directions.npy"
WALL_CENTER_PATH = DATA_ROOT / "wall_centers.npy"
TOPO_RESULTS_PATH = DATA_ROOT / "llama_topology_results.json"
OUTPUT_DIR = Path("/Users/ghost/Dev/hamster-ball/data")
OUTPUT_PLOT = OUTPUT_DIR / "phase6_convergence.png"

# Experiment hyper-parameters
LAMBDAS = [0.01, 0.1, 0.5, 1.0]
N_STEPS = 60
N_POINTS = 80          # points per synthetic cloud
AMBIENT_DIM = 20       # PCA-reduced dimensionality
SEED = 42
PERS_THRESHOLD = 0.15  # ignore beta_1 bars below this persistence (noise)

# ---------------------------------------------------------------------------
# Synthetic point-cloud generation
# ---------------------------------------------------------------------------

def _make_annular_cluster(
    center: np.ndarray,
    direction: np.ndarray,
    n_points: int,
    inner_r: float,
    outer_r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate points in an annular shell around *center* in the plane
    defined by *direction*, creating a beta_1 hole."""
    dim = len(center)
    # Build an orthonormal frame in the subspace perpendicular to direction
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    # Two random orthogonal vectors in the plane
    v1 = rng.standard_normal(dim)
    v1 -= v1.dot(direction) * direction
    v1 /= np.linalg.norm(v1) + 1e-12
    v2 = rng.standard_normal(dim)
    v2 -= v2.dot(direction) * direction
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2) + 1e-12

    angles = rng.uniform(0, 2 * np.pi, n_points)
    radii = rng.uniform(inner_r, outer_r, n_points)
    pts = center[None, :] + radii[:, None] * (
        np.cos(angles)[:, None] * v1[None, :] +
        np.sin(angles)[:, None] * v2[None, :]
    )
    # Small Gaussian jitter in all dimensions for realism
    pts += rng.normal(0, 0.05 * inner_r, pts.shape)
    return pts


def build_synthetic_cloud(
    wall_centers: np.ndarray,
    passage_dirs: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a point cloud that contains multiple beta_1 holes, calibrated
    to the wall geometry discovered in Phase 2."""
    n_walls = min(len(wall_centers), len(passage_dirs))
    pts_per_wall = max(n_points // n_walls, 12)
    remaining = n_points - pts_per_wall * n_walls

    all_pts = []
    for i in range(n_walls):
        c = wall_centers[i]
        d = passage_dirs[i]
        pts = _make_annular_cluster(
            center=c[:AMBIENT_DIM],
            direction=d[:AMBIENT_DIM],
            n_points=pts_per_wall,
            inner_r=3.0,
            outer_r=4.5,
            rng=rng,
        )
        all_pts.append(pts)

    # Fill remaining budget with background noise near the global centroid
    if remaining > 0:
        gc = np.mean(wall_centers[:n_walls, :AMBIENT_DIM], axis=0)
        noise = gc[None, :] + rng.normal(0, 3.0, (remaining, AMBIENT_DIM))
        all_pts.append(noise)

    return np.vstack(all_pts)

# ---------------------------------------------------------------------------
# Persistent-homology helpers
# ---------------------------------------------------------------------------

def compute_ph(points: np.ndarray, pers_threshold: float = PERS_THRESHOLD) -> dict:
    """Run Ripser and return beta counts + persistence info.

    Only beta_1 bars with persistence > pers_threshold are counted as
    genuine walls; anything below is topological noise.
    """
    res = ripser(points, maxdim=1, do_cocycles=True)
    dgm0 = res["dgms"][0]
    dgm1 = res["dgms"][1]
    cocycles1 = res["cocycles"][1]

    # beta_0: finite bars
    b0_finite = dgm0[np.isfinite(dgm0[:, 1])]
    beta_0 = len(b0_finite) + 1  # +1 for the infinite bar (connected component)

    # beta_1 bars (only above persistence threshold)
    beta_1_bars = []
    cycle_vertices = []
    for i, (birth, death) in enumerate(dgm1):
        if not np.isfinite(death):
            continue
        pers = death - birth
        if pers < pers_threshold:
            continue
        beta_1_bars.append((birth, death, pers))
        cocycle = cocycles1[i]
        verts = sorted(set(cocycle[:, :2].flatten().astype(int)))
        cycle_vertices.append(verts)

    beta_1_bars.sort(key=lambda t: t[2], reverse=True)

    max_pers = beta_1_bars[0][2] if beta_1_bars else 0.0
    total_pers = sum(t[2] for t in beta_1_bars)

    return {
        "beta_0": beta_0,
        "beta_1": len(beta_1_bars),
        "max_persistence": max_pers,
        "total_persistence": total_pers,
        "beta_1_bars": beta_1_bars,
        "cycle_vertices": cycle_vertices,
        "points": points,
    }

# ---------------------------------------------------------------------------
# Radial contraction step (simulated gradient of L_topology)
# ---------------------------------------------------------------------------

def radial_contraction_step(
    points: np.ndarray,
    cycle_vertices: list[list[int]],
    lam: float,
    lr: float = 0.15,
) -> np.ndarray:
    """Move cycle vertices toward their centroid, simulating
    grad-descent on L_topology = sum(persistence(beta_1 bars)).

    lambda scales the regularization strength; lr is the step size.
    """
    pts = points.copy()
    for verts in cycle_vertices:
        if len(verts) < 2:
            continue
        subset = pts[verts]
        centroid = subset.mean(axis=0)
        displacement = subset - centroid
        pts[verts] -= lam * lr * displacement
    return pts

# ---------------------------------------------------------------------------
# Run one full trajectory for a given lambda
# ---------------------------------------------------------------------------

def run_trajectory(
    cloud_init: np.ndarray,
    lam: float,
    n_steps: int,
) -> list[dict]:
    """Return per-step metrics for a single lambda trajectory."""
    pts = cloud_init.copy()
    history: list[dict] = []

    for step in range(n_steps + 1):
        ph = compute_ph(pts)
        history.append({
            "step": step,
            "lambda": lam,
            "beta_0": ph["beta_0"],
            "beta_1": ph["beta_1"],
            "max_persistence": ph["max_persistence"],
            "total_persistence": ph["total_persistence"],
        })
        if step < n_steps:
            pts = radial_contraction_step(
                pts,
                ph["cycle_vertices"],
                lam=lam,
            )
    return history

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(all_histories: dict[float, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Phase 6 — Wall Removal via Topology Regularization\n"
        r"Simulated gradient descent on $L_{\mathrm{topology}} = \Sigma\, "
        r"\mathrm{persistence}(\beta_1)$",
        fontsize=13,
    )

    colors = {0.01: "#1f77b4", 0.1: "#ff7f0e", 0.5: "#2ca02c", 1.0: "#d62728"}

    # (0,0) beta_1 count
    ax = axes[0, 0]
    for lam, hist in all_histories.items():
        steps = [h["step"] for h in hist]
        vals = [h["beta_1"] for h in hist]
        ax.plot(steps, vals, "-o", ms=3, label=f"$\\lambda$={lam}", color=colors[lam])
    ax.set_ylabel(r"$\beta_1$ count (walls)")
    ax.set_xlabel("Step")
    ax.set_title(r"Wall count ($\beta_1$) vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) max persistence
    ax = axes[0, 1]
    for lam, hist in all_histories.items():
        steps = [h["step"] for h in hist]
        vals = [h["max_persistence"] for h in hist]
        ax.plot(steps, vals, "-s", ms=3, label=f"$\\lambda$={lam}", color=colors[lam])
    ax.set_ylabel("Max persistence")
    ax.set_xlabel("Step")
    ax.set_title("Max wall strength vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) total persistence
    ax = axes[1, 0]
    for lam, hist in all_histories.items():
        steps = [h["step"] for h in hist]
        vals = [h["total_persistence"] for h in hist]
        ax.plot(steps, vals, "-^", ms=3, label=f"$\\lambda$={lam}", color=colors[lam])
    ax.set_ylabel(r"$\Sigma$ persistence")
    ax.set_xlabel("Step")
    ax.set_title(r"Total $\beta_1$ persistence vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) beta_0 stability
    ax = axes[1, 1]
    for lam, hist in all_histories.items():
        steps = [h["step"] for h in hist]
        vals = [h["beta_0"] for h in hist]
        ax.plot(steps, vals, "-d", ms=3, label=f"$\\lambda$={lam}", color=colors[lam])
    ax.set_ylabel(r"$\beta_0$ (connected components)")
    ax.set_xlabel("Step")
    ax.set_title(r"$\beta_0$ stability (should stay constant)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 6: Wall Removal Fine-Tuning PoC")
    print("  Topology regularization simulation")
    print("=" * 70)

    # Load Phase-2 artifacts
    passage_dirs = np.load(PASSAGE_DIR_PATH)
    wall_centers = np.load(WALL_CENTER_PATH)
    with open(TOPO_RESULTS_PATH) as f:
        topo_results = json.load(f)

    print(f"\n  Loaded passage_directions : {passage_dirs.shape}")
    print(f"  Loaded wall_centers       : {wall_centers.shape}")
    n_prompts = len(topo_results)
    avg_beta1 = np.mean([v["beta_1"] for v in topo_results.values()])
    avg_wall_str = np.mean([v["wall_strength"] for v in topo_results.values()])
    print(f"  Prompts in topology data  : {n_prompts}")
    print(f"  Average beta_1 count      : {avg_beta1:.1f}")
    print(f"  Average wall strength     : {avg_wall_str:.2f}")

    # Build a single deterministic synthetic cloud for fair comparison
    rng = np.random.default_rng(SEED)
    cloud_init = build_synthetic_cloud(
        wall_centers, passage_dirs, N_POINTS, rng
    )
    print(f"\n  Synthetic cloud shape     : {cloud_init.shape}")

    # Baseline PH
    ph0 = compute_ph(cloud_init)
    print(f"  Initial beta_0            : {ph0['beta_0']}")
    print(f"  Initial beta_1            : {ph0['beta_1']}")
    print(f"  Initial max persistence   : {ph0['max_persistence']:.4f}")
    print(f"  Initial total persistence : {ph0['total_persistence']:.4f}")

    # Run trajectories for each lambda
    all_histories: dict[float, list[dict]] = {}
    for lam in LAMBDAS:
        print(f"\n  --- lambda = {lam} ---")
        t0 = time.time()
        hist = run_trajectory(cloud_init, lam, N_STEPS)
        dt = time.time() - t0
        final = hist[-1]
        print(f"    Steps: {N_STEPS} in {dt:.1f}s")
        print(f"    Final beta_1        : {final['beta_1']}")
        print(f"    Final max persist.  : {final['max_persistence']:.4f}")
        print(f"    Final total persist.: {final['total_persistence']:.4f}")
        print(f"    beta_0 drift        : {hist[0]['beta_0']} -> {final['beta_0']}")

        # Steps to reach beta_1 = 0
        zero_steps = [h["step"] for h in hist if h["beta_1"] == 0]
        if zero_steps:
            print(f"    beta_1 = 0 at step  : {zero_steps[0]}")
        else:
            print(f"    beta_1 never reached 0 (min: {min(h['beta_1'] for h in hist)})")

        all_histories[lam] = hist

    # Plot
    print("\n  Generating convergence plot...")
    plot_convergence(all_histories, OUTPUT_PLOT)

    # Summary table
    print("\n" + "=" * 70)
    print("  CONVERGENCE SUMMARY")
    print("=" * 70)
    print(f"  {'lambda':>8s}  {'steps_to_0':>11s}  {'final_b1':>9s}  "
          f"{'final_max_p':>12s}  {'b0_stable':>10s}")
    print("  " + "-" * 60)
    for lam in LAMBDAS:
        hist = all_histories[lam]
        final = hist[-1]
        zero_steps = [h["step"] for h in hist if h["beta_1"] == 0]
        steps_str = str(zero_steps[0]) if zero_steps else "never"
        b0_start, b0_end = hist[0]["beta_0"], final["beta_0"]
        stable = "YES" if abs(b0_start - b0_end) <= 2 else "NO"
        print(f"  {lam:>8.2f}  {steps_str:>11s}  {final['beta_1']:>9d}  "
              f"{final['max_persistence']:>12.4f}  {stable:>10s}")
    print("=" * 70)
    print("\n  Phase 6 PoC complete.")


if __name__ == "__main__":
    main()
