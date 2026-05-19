from src.multiple_runs import MultiDW
import numpy as np
import matplotlib.pyplot as plt

def test_entropy_std():
    ds = np.linspace(0.05, 0.5, 10)
    mus = np.linspace(0.05, 0.5, 10)
    snapshots = [4, 24, 49]
    entropies = np.zeros((len(snapshots), len(ds), len(mus)))
    stds = np.zeros((len(snapshots), len(ds), len(mus)))
    clusters = np.zeros((len(snapshots), len(ds), len(mus)))
    for row, mu in enumerate(mus):
        for col, d in enumerate(ds):
            cal = MultiDW(
                N=1000,
                d=d,
                mu=mu,
                t=int(10 * (3 + 1 / mu)),
                topology="full",
                num_of_runs=1,
                snapshots=snapshots,
            )
            stats = cal.run()
            std, cluster, kde, hist, entropy = stats
            entropies[:, row, col] = entropy
            stds[:, row, col] = std
            clusters[:, row, col] = cluster
            print(f"Completed d={d}, mu={mu}")
    
    data_step = ds[1] - ds[0]  # Spacing between data points
    extent_min = ds[0] - data_step / 2  # Extend half a step on each side
    extent_max = ds[-1] + data_step / 2
    
    for i, snapshot in enumerate(snapshots):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        im = ax.imshow(entropies[i], extent=(extent_min, extent_max, extent_min, extent_max), origin="lower", aspect="auto")
        ax.set_xticks(ds)
        ax.set_xticklabels([f"{val:.2f}" for val in ds])
        ax.set_yticks(mus)
        ax.set_yticklabels([f"{val:.2f}" for val in mus])
        plt.colorbar(im, label="Entropy")
        plt.title(f"Entropy at t = {snapshot}")
        plt.xlabel("d")
        plt.ylabel("mu")
        plt.tight_layout()
        plt.savefig(f"results/entropy_{snapshot}.png")

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        im = ax.imshow(stds[i], extent=(extent_min, extent_max, extent_min, extent_max), origin="lower", aspect="auto")
        ax.set_xticks(ds)
        ax.set_xticklabels([f"{val:.2f}" for val in ds])
        ax.set_yticks(mus)
        ax.set_yticklabels([f"{val:.2f}" for val in mus])
        plt.colorbar(im, label="Standard Deviation")
        plt.title(f"Standard Deviation at t = {snapshot}")
        plt.xlabel("d")
        plt.ylabel("mu")
        plt.tight_layout()
        plt.savefig(f"results/std_{snapshot}.png")

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        im = ax.imshow(clusters[i], extent=(extent_min, extent_max, extent_min, extent_max), origin="lower", aspect="auto")
        ax.set_xticks(ds)
        ax.set_xticklabels([f"{val:.2f}" for val in ds])
        ax.set_yticks(mus)
        ax.set_yticklabels([f"{val:.2f}" for val in mus])
        plt.colorbar(im, label="Number of Clusters")
        plt.title(f"Number of Clusters at t = {snapshot+1}")
        plt.xlabel("d")
        plt.ylabel("mu")
        plt.tight_layout()
        plt.savefig(f"results/clusters_{snapshot+1}.png")


def kde_difference_heatmap(d_ref, mu_ref):
    """
    Create heatmaps showing the sum of absolute differences in KDE
    between the model with reference parameters (d_ref, mu_ref) and
    each grid point in the d-mu parameter space.
    
    Args:
        d_ref: Reference d parameter
        mu_ref: Reference mu parameter
    """
    ds = np.linspace(0.05, 0.5, 10)
    mus = np.linspace(0.05, 0.5, 10)
    snapshots = [49]
    
    # Get reference KDE
    cal_ref = MultiDW(
        N=1000,
        d=d_ref,
        mu=mu_ref,
        t=int(10 * (3 + 1 / mu_ref)),
        topology="full",
        num_of_runs=5,
        snapshots=snapshots,
    )
    stats_ref = cal_ref.run()
    _, _, kde_ref, hist_ref, _ = stats_ref
    kde_ref = np.array(kde_ref[0])  # Get first snapshot
    print(f"Reference model (d={d_ref}, mu={mu_ref}) completed")
    
    # Create heatmap for KDE differences
    kde_diffs = np.zeros((len(ds), len(mus)))
    hist_diffs = np.zeros((len(ds), len(mus)))
    
    for row, mu in enumerate(mus):
        for col, d in enumerate(ds):
            cal = MultiDW(
                N=1000,
                d=d,
                mu=mu,
                t=int(10 * (3 + 1 / mu)),
                topology="full",
                num_of_runs=5,
                snapshots=snapshots,
            )
            stats = cal.run()
            _, _, kde, hist, _ = stats
            kde = np.array(kde[0])  # Get first snapshot
            hist = np.array(hist[0])  # Get first snapshot
            
            # Calculate sum of absolute differences in kde
            kde_diffs[row, col] = np.sum(np.abs(kde - kde_ref))
            
            # Calculate sum of absolute differences in histogram
            hist_diffs[row, col] = np.sum(np.abs(hist - hist_ref))
            
            print(f"Completed d={d}, mu={mu}")
    
    # Calculate extent to align pixels with data points
    data_step = ds[1] - ds[0]
    extent_min = ds[0] - data_step / 2
    extent_max = ds[-1] + data_step / 2
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(kde_diffs, extent=(extent_min, extent_max, extent_min, extent_max), origin="lower", aspect="auto")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{val:.2f}" for val in ds])
    ax.set_yticks(mus)
    ax.set_yticklabels([f"{val:.2f}" for val in mus])
    plt.colorbar(im, label="Sum of KDE Differences")
    plt.title(f"Sum of KDE Differences from (d={d_ref}, mu={mu_ref})")
    plt.xlabel("d")
    plt.ylabel("mu")
    plt.tight_layout()
    plt.savefig(f"results/kde_diff_heatmap_d{d_ref}_mu{mu_ref}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(hist_diffs, extent=(extent_min, extent_max, extent_min, extent_max), origin="lower", aspect="auto")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{val:.2f}" for val in ds])
    ax.set_yticks(mus)
    ax.set_yticklabels([f"{val:.2f}" for val in mus])
    plt.colorbar(im, label="Sum of Histogram Differences")
    plt.title(f"Sum of Histogram Differences from (d={d_ref}, mu={mu_ref})")
    plt.xlabel("d")
    plt.ylabel("mu")
    plt.tight_layout()
    plt.savefig(f"results/hist_diff_heatmap_d{d_ref}_mu{mu_ref}.png")
    plt.close()


if __name__ == "__main__":
    # test_entropy_std()
    kde_difference_heatmap(d_ref=0.1, mu_ref=0.1)
    kde_difference_heatmap(d_ref=0.4, mu_ref=0.4)