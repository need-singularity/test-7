"""
Neuron Layout: 4096 neurons of Llama 8B arranged in 2D/3D by passage direction correlation.

Visualizes which neurons cluster together as boundary-forming dimensions,
particularly dim 940 and 1917 which recur across multiple prompts.
"""

import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_passage_directions(data_dir: Path) -> np.ndarray:
    """Load passage direction vectors from Phase 2."""
    path = data_dir / "passage_directions.npy"
    directions = np.load(path)
    print(f"Loaded {directions.shape[0]} passage directions, dim={directions.shape[1]}")
    return directions


def compute_neuron_profiles(directions: np.ndarray) -> np.ndarray:
    """
    Build a profile for each neuron (dim) across all walls.

    directions: shape (n_walls, 4096)
    Returns: shape (4096, n_walls) — each row is a neuron's contribution across walls.
    """
    return directions.T  # (4096, n_walls)


def layout_2d(profiles: np.ndarray, top_dims: list[int], method: str = "tsne"):
    """Arrange 4096 neurons in 2D based on their wall-contribution profiles."""
    print(f"\n2D layout ({method})...")

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        reducer = PCA(n_components=2)

    coords_2d = reducer.fit_transform(profiles)

    # Magnitude of each neuron across all walls
    magnitudes = np.linalg.norm(profiles, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # All neurons (faint)
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=magnitudes, cmap='viridis', s=3, alpha=0.3,
        label='all neurons'
    )
    plt.colorbar(scatter, ax=ax, label='Wall contribution magnitude')

    # Highlight top dims
    colors = ['red', 'orange', 'magenta', 'cyan', 'lime']
    for i, dim in enumerate(top_dims):
        ax.scatter(
            coords_2d[dim, 0], coords_2d[dim, 1],
            c=colors[i % len(colors)], s=200, marker='*',
            edgecolors='black', linewidths=1.5, zorder=5,
            label=f'dim {dim}'
        )
        ax.annotate(
            f'dim {dim}', (coords_2d[dim, 0], coords_2d[dim, 1]),
            fontsize=11, fontweight='bold',
            xytext=(10, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black')
        )

    # Check if 940 and 1917 are close
    if 940 in top_dims and 1917 in top_dims:
        d = np.linalg.norm(coords_2d[940] - coords_2d[1917])
        median_dist = np.median(np.linalg.norm(
            coords_2d[np.random.choice(4096, 1000)] - coords_2d[np.random.choice(4096, 1000)],
            axis=1
        ))
        ax.set_title(
            f'Neuron Layout (2D {method.upper()})\n'
            f'dim940↔1917 distance: {d:.2f} (median random: {median_dist:.2f})',
            fontsize=14
        )
    else:
        ax.set_title(f'Neuron Layout (2D {method.upper()})', fontsize=14)

    ax.legend(loc='upper right')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    out_path = Path(__file__).parent.parent / "data" / f"neuron_layout_2d_{method}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    return coords_2d


def layout_3d(profiles: np.ndarray, top_dims: list[int], method: str = "tsne"):
    """Arrange 4096 neurons in 3D."""
    print(f"\n3D layout ({method})...")

    if method == "tsne":
        reducer = TSNE(n_components=3, perplexity=30, random_state=42)
    else:
        reducer = PCA(n_components=3)

    coords_3d = reducer.fit_transform(profiles)
    magnitudes = np.linalg.norm(profiles, axis=1)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # All neurons
    scatter = ax.scatter(
        coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
        c=magnitudes, cmap='viridis', s=2, alpha=0.2
    )

    # Highlight top dims
    colors = ['red', 'orange', 'magenta', 'cyan', 'lime']
    for i, dim in enumerate(top_dims):
        ax.scatter(
            coords_3d[dim, 0], coords_3d[dim, 1], coords_3d[dim, 2],
            c=colors[i % len(colors)], s=300, marker='*',
            edgecolors='black', linewidths=1.5, zorder=5,
            label=f'dim {dim}'
        )
        ax.text(
            coords_3d[dim, 0], coords_3d[dim, 1], coords_3d[dim, 2],
            f'  dim {dim}', fontsize=10, fontweight='bold'
        )

    ax.set_title(f'Neuron Layout (3D {method.upper()})', fontsize=14)
    ax.legend(loc='upper right')

    out_path = Path(__file__).parent.parent / "data" / f"neuron_layout_3d_{method}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    return coords_3d


def analyze_clusters(profiles: np.ndarray, coords_2d: np.ndarray, top_dims: list[int]):
    """Analyze whether top dims form a cluster."""
    print("\n--- Cluster Analysis ---")

    top_coords = coords_2d[top_dims]

    # Pairwise distances among top dims
    print("\nPairwise distances (top dims):")
    for i in range(len(top_dims)):
        for j in range(i+1, len(top_dims)):
            d = np.linalg.norm(top_coords[i] - top_coords[j])
            print(f"  dim {top_dims[i]} ↔ dim {top_dims[j]}: {d:.2f}")

    # Compare to random baseline
    n_samples = 10000
    idx_a = np.random.choice(4096, n_samples)
    idx_b = np.random.choice(4096, n_samples)
    random_dists = np.linalg.norm(coords_2d[idx_a] - coords_2d[idx_b], axis=1)

    top_mean_dist = np.mean([
        np.linalg.norm(top_coords[i] - top_coords[j])
        for i in range(len(top_dims))
        for j in range(i+1, len(top_dims))
    ])

    percentile = np.mean(random_dists > top_mean_dist) * 100

    print(f"\nTop dims mean distance: {top_mean_dist:.2f}")
    print(f"Random mean distance:   {np.mean(random_dists):.2f}")
    print(f"Top dims are closer than {percentile:.1f}% of random pairs")

    if percentile > 50:
        print("→ Top dims are CLUSTERED (closer than average)")
    else:
        print("→ Top dims are SPREAD (not clustered)")


def main():
    data_dir = Path(__file__).parent.parent.parent / "test-7" / "data" / "poc_topology"

    if not (data_dir / "passage_directions.npy").exists():
        print(f"passage_directions.npy not found at {data_dir}")
        print("Run Phase 2 first: python experiments/phase2_hole_directions.py")
        return

    directions = load_passage_directions(data_dir)
    profiles = compute_neuron_profiles(directions)

    # Key dims from Phase 2 results
    top_dims = [940, 1917, 2977, 3884, 2720, 866, 133, 3951, 406, 3433]

    # 2D layouts
    coords_2d_tsne = layout_2d(profiles, top_dims, method="tsne")
    layout_2d(profiles, top_dims, method="pca")

    # 3D layouts (need at least 3 walls for 3D t-SNE)
    n_walls = directions.shape[0]
    if n_walls >= 3:
        layout_3d(profiles, top_dims, method="tsne")
        layout_3d(profiles, top_dims, method="pca")
    else:
        print(f"\nSkipping 3D layout: only {n_walls} walls (need ≥3)")
        layout_2d(profiles, top_dims, method="pca")  # fallback

    # Cluster analysis
    analyze_clusters(profiles, coords_2d_tsne, top_dims)


if __name__ == "__main__":
    main()
