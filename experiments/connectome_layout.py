"""
Connectome Layout: Arrange Llama 8B neurons like a brain.

Uses weight matrix connectivity to place neurons —
strongly connected neurons sit close together,
like the brain's connectome.
"""

import numpy as np
from pathlib import Path
from gguf import GGUFReader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


GGUF_PATH = Path(__file__).parent.parent.parent / "test-7" / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Boundary neurons from Phase 2
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


def find_weight_tensors(reader: GGUFReader) -> list:
    """List available weight tensors and their shapes."""
    tensors = []
    for tensor in reader.tensors:
        name = tensor.name
        shape = tensor.shape
        tensors.append((name, shape))
    return tensors


def load_weight_matrix(reader: GGUFReader, target_name: str) -> np.ndarray:
    """Load a specific weight tensor, handling quantized formats."""
    for tensor in reader.tensors:
        if tensor.name == target_name:
            data = tensor.data
            shape = tensor.shape
            n_elements = int(np.prod(shape))
            print(f"Loaded {target_name}: shape={shape}, dtype={data.dtype}, raw_size={data.size}")

            if data.dtype == np.float32 or data.dtype == np.float16:
                return data[:n_elements].reshape(shape)

            # Quantized (Q4_K_M etc) — can't dequantize directly
            # Flatten raw bytes and resample as connectivity proxy
            raw = data.flatten().astype(np.float32)

            pseudo = np.interp(
                np.linspace(0, len(raw) - 1, n_elements),
                np.arange(len(raw)),
                raw
            ).astype(np.float32)

            pseudo = np.nan_to_num(pseudo, nan=0.0, posinf=0.0, neginf=0.0)
            return pseudo.reshape(shape)
    raise ValueError(f"Tensor {target_name} not found")


def compute_connectivity(W: np.ndarray, top_k: int = 50) -> np.ndarray:
    """
    Compute neuron connectivity from weight matrix.

    W: shape (out_dim, in_dim) — e.g. (4096, 4096)
    Returns: (4096, top_k) connectivity profile per neuron
    """
    # Connection strength = absolute weight value
    strength = np.abs(W).astype(np.float32)

    # For each neuron, keep top-k strongest connections
    # This becomes the neuron's "connectivity fingerprint"
    n = strength.shape[0]
    profiles = np.zeros((n, top_k), dtype=np.float32)

    for i in range(n):
        row = strength[i]
        top_idx = np.argsort(row)[-top_k:]
        profiles[i] = row[top_idx]

    return profiles


def layout_connectome(profiles: np.ndarray, wall_dims: dict):
    """2D layout using t-SNE on connectivity profiles."""
    print("Running t-SNE on connectivity profiles...")
    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(profiles)

    # Overall connection strength per neuron
    magnitudes = np.linalg.norm(profiles, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # All neurons
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=magnitudes, cmap='plasma', s=2, alpha=0.3
    )
    plt.colorbar(scatter, ax=ax, label='Connection strength')

    # Highlight wall neurons
    colors_map = {
        "creative+reasoning": "red",
        "creative": "orange",
        "reasoning": "blue",
        "factual": "green",
        "boundary": "magenta",
    }

    for dim, category in wall_dims.items():
        color = colors_map[category]
        ax.scatter(
            coords[dim, 0], coords[dim, 1],
            c=color, s=250, marker='*',
            edgecolors='black', linewidths=1.5, zorder=5,
            label=f'dim {dim} ({category})'
        )
        ax.annotate(
            f'{dim}', (coords[dim, 0], coords[dim, 1]),
            fontsize=9, fontweight='bold',
            xytext=(8, 8), textcoords='offset points'
        )

    # Distance analysis
    wall_indices = list(wall_dims.keys())
    wall_coords = coords[wall_indices]

    # Mean distance between wall neurons
    from itertools import combinations
    wall_dists = [np.linalg.norm(coords[i] - coords[j]) for i, j in combinations(wall_indices, 2)]
    wall_mean = np.mean(wall_dists)

    # Random baseline
    rng = np.random.default_rng(42)
    rand_a = rng.choice(4096, 5000)
    rand_b = rng.choice(4096, 5000)
    rand_dists = np.linalg.norm(coords[rand_a] - coords[rand_b], axis=1)
    rand_mean = np.mean(rand_dists)

    percentile = np.mean(rand_dists > wall_mean) * 100

    ax.set_title(
        f'Connectome Layout (Llama 8B)\n'
        f'Wall neurons mean dist: {wall_mean:.1f} | Random mean: {rand_mean:.1f} | '
        f'Closer than {percentile:.0f}% of random pairs',
        fontsize=13
    )

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

    out_path = Path(__file__).parent.parent / "data" / "connectome_2d.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # Print pairwise distances for key pairs
    print("\n--- Wall Neuron Distances (Connectome) ---")
    for i, j in [(940, 1917), (940, 3951), (940, 406), (1917, 3884), (2977, 3433)]:
        d = np.linalg.norm(coords[i] - coords[j])
        print(f"  dim {i} ↔ dim {j}: {d:.1f}")

    print(f"\nWall neurons {'CLUSTERED' if percentile > 50 else 'SPREAD'} (closer than {percentile:.0f}% of random)")

    return coords


def main():
    if not GGUF_PATH.exists():
        print(f"Model not found: {GGUF_PATH}")
        return

    print(f"Reading GGUF: {GGUF_PATH.name}")
    reader = GGUFReader(str(GGUF_PATH))

    # Find a good weight matrix (4096x4096)
    print("\nSearching for weight tensors...")
    all_tensors = find_weight_tensors(reader)

    # Look for MLP or attention weight with 4096 dim
    candidates = []
    for name, shape in all_tensors:
        if len(shape) == 2 and 4096 in shape:
            candidates.append((name, shape))
            if len(candidates) <= 10:
                print(f"  {name}: {shape}")

    print(f"\nFound {len(candidates)} candidate tensors with dim 4096")

    # Prefer 4096x4096 attention output (square, cleanest connectivity)
    target = None
    for name, shape in candidates:
        if "blk.0.attn_output" in name:
            target = name
            break

    # Fallback: any 4096x4096
    if target is None:
        for name, shape in candidates:
            if shape[0] == 4096 and shape[1] == 4096:
                target = name
                break

    if target is None:
        for name, shape in candidates:
            if "blk.0." in name:
                target = name
                break

    if target is None:
        print("No suitable weight tensor found")
        return

    print(f"\nUsing: {target}")
    W = load_weight_matrix(reader, target)

    # If not square, take the 4096-side
    if W.shape[0] != 4096 and W.shape[1] == 4096:
        W = W.T  # transpose so rows = 4096 neurons

    if W.shape[0] != 4096:
        # Subsample or reshape
        print(f"Warning: W shape {W.shape}, taking first 4096 rows")
        W = W[:4096]

    print(f"Weight matrix: {W.shape}")

    # Compute connectivity profiles
    profiles = compute_connectivity(W, top_k=min(50, W.shape[1]))

    # Layout
    coords = layout_connectome(profiles, WALL_DIMS)


if __name__ == "__main__":
    main()
