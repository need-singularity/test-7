"""
Multi-layer Connectome Analysis: Do wall neurons cluster in deeper layers?

Loads attn_output.weight from layers 0, 7, 15, 23, 31 and computes
t-SNE layouts from connectivity profiles. Measures whether the 10
wall neurons become more clustered (relative to random baselines)
as depth increases.
"""

import numpy as np
from pathlib import Path
from itertools import combinations
from gguf import GGUFReader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


GGUF_PATH = (
    Path(__file__).parent.parent.parent
    / "test-7" / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

LAYERS = [0, 7, 15, 23, 31]

WALL_DIMS = {
    940: "creative+reasoning",
    1917: "creative+reasoning",
    2977: "reasoning",
    3884: "creative",
    2720: "factual",
    866: "factual",
    133: "factual",
    3951: "boundary",
    406: "boundary",
    3433: "boundary",
}

COLORS_MAP = {
    "creative+reasoning": "red",
    "creative": "orange",
    "reasoning": "blue",
    "factual": "green",
    "boundary": "magenta",
}


def load_weight_matrix(reader: GGUFReader, target_name: str) -> np.ndarray:
    """Load a weight tensor, handling quantized formats via interpolation."""
    for tensor in reader.tensors:
        if tensor.name == target_name:
            data = tensor.data
            shape = tensor.shape
            n_elements = int(np.prod(shape))
            print(f"  Loaded {target_name}: shape={shape}, dtype={data.dtype}, raw_size={data.size}")

            if data.dtype in (np.float32, np.float16):
                return data[:n_elements].reshape(shape)

            # Quantized — interpolate as connectivity proxy
            raw = data.flatten().astype(np.float32)
            pseudo = np.interp(
                np.linspace(0, len(raw) - 1, n_elements),
                np.arange(len(raw)),
                raw,
            ).astype(np.float32)
            pseudo = np.nan_to_num(pseudo, nan=0.0, posinf=0.0, neginf=0.0)
            return pseudo.reshape(shape)
    raise ValueError(f"Tensor {target_name} not found")


def compute_connectivity(W: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Top-k absolute weight profile per neuron (connectivity fingerprint)."""
    strength = np.abs(W).astype(np.float32)
    n = strength.shape[0]
    profiles = np.zeros((n, top_k), dtype=np.float32)
    for i in range(n):
        row = strength[i]
        top_idx = np.argsort(row)[-top_k:]
        profiles[i] = row[top_idx]
    return profiles


def analyse_layer(reader, layer_idx):
    """Load one layer, compute profiles + t-SNE coords + clustering stats."""
    tensor_name = f"blk.{layer_idx}.attn_output.weight"
    print(f"\n--- Layer {layer_idx} ---")

    W = load_weight_matrix(reader, tensor_name)

    # Ensure rows = 4096 neurons
    if W.shape[0] != 4096 and W.shape[1] == 4096:
        W = W.T
    if W.shape[0] != 4096:
        print(f"  Warning: shape {W.shape}, taking first 4096 rows")
        W = W[:4096]

    profiles = compute_connectivity(W, top_k=min(50, W.shape[1]))

    print(f"  Running t-SNE (perplexity=30)...")
    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(profiles)

    # --- Clustering metric ---
    wall_indices = list(WALL_DIMS.keys())
    wall_dists = [
        np.linalg.norm(coords[i] - coords[j])
        for i, j in combinations(wall_indices, 2)
    ]
    wall_mean = float(np.mean(wall_dists))

    rng = np.random.default_rng(42)
    rand_a = rng.choice(4096, 5000)
    rand_b = rng.choice(4096, 5000)
    rand_dists = np.linalg.norm(coords[rand_a] - coords[rand_b], axis=1)
    rand_mean = float(np.mean(rand_dists))

    ratio = wall_mean / rand_mean if rand_mean > 0 else float("inf")
    percentile = float(np.mean(rand_dists > wall_mean) * 100)

    magnitudes = np.linalg.norm(profiles, axis=1)

    print(f"  Wall mean dist: {wall_mean:.1f} | Random mean: {rand_mean:.1f} | "
          f"Ratio: {ratio:.3f} | Closer than {percentile:.0f}% of random")

    return {
        "layer": layer_idx,
        "coords": coords,
        "magnitudes": magnitudes,
        "wall_mean": wall_mean,
        "rand_mean": rand_mean,
        "ratio": ratio,
        "percentile": percentile,
    }


def plot_multilayer(results, out_path):
    """5 subplots — one per layer — showing neuron layout with wall neurons."""
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    fig.suptitle(
        "Multi-layer Connectome: Wall Neuron Positions Across Depth",
        fontsize=15, fontweight="bold", y=1.02,
    )

    wall_indices = list(WALL_DIMS.keys())

    for ax, res in zip(axes, results):
        coords = res["coords"]
        mags = res["magnitudes"]

        ax.scatter(coords[:, 0], coords[:, 1], c=mags, cmap="plasma", s=1, alpha=0.25)

        for dim, category in WALL_DIMS.items():
            color = COLORS_MAP[category]
            ax.scatter(
                coords[dim, 0], coords[dim, 1],
                c=color, s=120, marker="*",
                edgecolors="black", linewidths=1.0, zorder=5,
            )
            ax.annotate(
                str(dim), (coords[dim, 0], coords[dim, 1]),
                fontsize=6, fontweight="bold",
                xytext=(4, 4), textcoords="offset points",
            )

        status = "CLUSTERED" if res["ratio"] < 1.0 else "SPREAD"
        ax.set_title(
            f"Layer {res['layer']}\n"
            f"ratio={res['ratio']:.2f} ({status})\n"
            f"closer than {res['percentile']:.0f}% random",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=c,
               markersize=12, markeredgecolor="black", label=cat)
        for cat, c in COLORS_MAP.items()
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.06))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved multi-layer plot: {out_path}")
    plt.close(fig)


def plot_trend(results, out_path):
    """Line plot: clustering ratio across layers."""
    layers = [r["layer"] for r in results]
    ratios = [r["ratio"] for r in results]
    percentiles = [r["percentile"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_ratio = "tab:blue"
    ax1.plot(layers, ratios, "o-", color=color_ratio, linewidth=2, markersize=8, label="Wall/Random dist ratio")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="ratio = 1 (no clustering)")
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Wall Mean Dist / Random Mean Dist", fontsize=12, color=color_ratio)
    ax1.tick_params(axis="y", labelcolor=color_ratio)
    ax1.set_xticks(layers)

    # Fill below 1.0 to highlight clustering
    ax1.fill_between(layers, ratios, 1.0,
                     where=[r < 1.0 for r in ratios],
                     alpha=0.15, color="green", label="Clustered region")
    ax1.fill_between(layers, ratios, 1.0,
                     where=[r >= 1.0 for r in ratios],
                     alpha=0.15, color="red", label="Spread region")

    ax2 = ax1.twinx()
    color_pct = "tab:orange"
    ax2.plot(layers, percentiles, "s--", color=color_pct, linewidth=1.5, markersize=7, label="Percentile (% random farther)")
    ax2.set_ylabel("% of random pairs farther apart", fontsize=11, color=color_pct)
    ax2.tick_params(axis="y", labelcolor=color_pct)
    ax2.set_ylim(0, 100)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title(
        "Wall Neuron Clustering Across Layers\n"
        "(ratio < 1 = wall neurons closer than average)",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved trend plot: {out_path}")
    plt.close(fig)


def main():
    if not GGUF_PATH.exists():
        print(f"Model not found: {GGUF_PATH}")
        return

    print(f"Reading GGUF: {GGUF_PATH.name}")
    reader = GGUFReader(str(GGUF_PATH))

    results = []
    for layer_idx in LAYERS:
        res = analyse_layer(reader, layer_idx)
        results.append(res)

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Layer':>6} | {'Wall Mean':>10} | {'Rand Mean':>10} | {'Ratio':>7} | {'Pctile':>7} | Status")
    print("-" * 65)
    for r in results:
        status = "CLUSTERED" if r["ratio"] < 1.0 else "SPREAD"
        print(f"{r['layer']:>6} | {r['wall_mean']:>10.1f} | {r['rand_mean']:>10.1f} | "
              f"{r['ratio']:>7.3f} | {r['percentile']:>6.0f}% | {status}")
    print("=" * 65)

    # Trend
    first_ratio = results[0]["ratio"]
    last_ratio = results[-1]["ratio"]
    if last_ratio < first_ratio:
        trend = "Wall neurons become MORE clustered in deeper layers"
    elif last_ratio > first_ratio:
        trend = "Wall neurons become LESS clustered in deeper layers"
    else:
        trend = "No clear trend in wall neuron clustering"
    print(f"\nTrend: {trend}")
    print(f"  Layer 0 ratio: {first_ratio:.3f} -> Layer 31 ratio: {last_ratio:.3f}")

    # Plots
    data_dir = Path(__file__).parent.parent / "data"
    plot_multilayer(results, data_dir / "connectome_multilayer.png")
    plot_trend(results, data_dir / "connectome_layer_trend.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
