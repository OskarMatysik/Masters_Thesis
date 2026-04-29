import matplotlib.pyplot as plt
import numpy as np

from models import DeffuantWeisbuchModel


def run_parameter_sweep(
    N: int = 1000, topology: str = "full"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DeffuantWeisbuchModel for each pair of parameters d and mu.
    Parameters range from 0 to 0.5 with 0.05 step.

    Args:
        N: Number of agents
        topology: Topology of the network
        t: Number of time steps (None for convergence-based stopping)

    Returns:
        Tuple of (parameters array, entropy_matrix, std_matrix)
        where parameters are in shape (11,) for values [0, 0.05, ..., 0.5]
    """
    params = np.arange(0.05, 0.55, 0.05)
    entropy_matrix = np.zeros((len(params), len(params)))
    std_matrix = np.zeros((len(params), len(params)))

    total_runs = len(params) ** 2
    run_count = 0

    for i, d in enumerate(params):
        for j, mu in enumerate(params):
            run_count += 1
            print(f"Running {run_count}/{total_runs}: d={d:.2f}, mu={mu:.2f}")
            t = int(10 * (3 + 1 / mu))
            model = DeffuantWeisbuchModel(N=N, d=d, mu=mu, t=t, topology=topology)
            model.run()
            std, _, _, entropy = model.statistics()

            entropy_matrix[i, j] = entropy
            std_matrix[i, j] = std

    return params, entropy_matrix, std_matrix


def plot_heatmaps(
    params: np.ndarray,
    entropy_matrix: np.ndarray,
    std_matrix: np.ndarray,
    save_path: str = "results/heatmaps.png",
) -> None:
    """
    Create two heatmaps: one for entropy and one for standard deviation.

    Args:
        params: Array of parameter values used
        entropy_matrix: 2D array of entropy values (d x mu)
        std_matrix: 2D array of standard deviation values (d x mu)
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Entropy heatmap
    im1 = axes[0].imshow(entropy_matrix, cmap="viridis", aspect="auto", origin="lower")
    axes[0].set_xlabel("μ (mu)")
    axes[0].set_ylabel("d (disagreement threshold)")
    axes[0].set_title("Entropy Heatmap")
    axes[0].set_xticks(range(len(params)))
    axes[0].set_yticks(range(len(params)))
    axes[0].set_xticklabels([f"{p:.2f}" for p in params], rotation=45)
    axes[0].set_yticklabels([f"{p:.2f}" for p in params])
    plt.colorbar(im1, ax=axes[0])

    # Standard Deviation heatmap
    im2 = axes[1].imshow(std_matrix, cmap="plasma", aspect="auto", origin="lower")
    axes[1].set_xlabel("μ (mu)")
    axes[1].set_ylabel("d (disagreement threshold)")
    axes[1].set_title("Standard Deviation Heatmap")
    axes[1].set_xticks(range(len(params)))
    axes[1].set_yticks(range(len(params)))
    axes[1].set_xticklabels([f"{p:.2f}" for p in params], rotation=45)
    axes[1].set_yticklabels([f"{p:.2f}" for p in params])
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Heatmaps saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Run parameter sweep
    params, entropy_matrix, std_matrix = run_parameter_sweep(N=1000, topology="full")

    # Plot heatmaps
    plot_heatmaps(params, entropy_matrix, std_matrix)
