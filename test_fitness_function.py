from src.multiple_runs import MultiDW
from src.polarization import esteban_ray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks


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
            std, cluster, kde, hist, entropy, observations = stats
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
    std, cluster_count, kde_ref, hist_ref, entropy, observations = stats_ref
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
            std, cluster_count, kde, hist, entropy, observations = stats
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


def wasserstein_distance_heatmap(d_ref, mu_ref, snapshot):
    """
    Create heatmaps showing the Wasserstein distance
    between the model with reference parameters (d_ref, mu_ref) and
    each grid point in the d-mu parameter space.
    
    Args:
        d_ref: Reference d parameter
        mu_ref: Reference mu parameter
    """
    ds = np.linspace(0.05, 0.5, 10)
    mus = np.linspace(0.05, 0.5, 10)
    snapshots = [snapshot]
    
    # Get reference observations
    cal_ref = MultiDW(
        N=1000,
        d=d_ref,
        mu=mu_ref,
        t=snapshot+1,
        topology="full",
        num_of_runs=1,
        snapshots=snapshots,
    )
    stats_ref = cal_ref.run()
    std, cluster_count, kde_ref, hist_ref, entropy, observations_ref = stats_ref
    ref_obs = np.array(observations_ref[0][0])  # Get first run, first snapshot
    peaks_ref = len(find_peaks(ref_obs)[0])
    print(f"Reference model (d={d_ref}, mu={mu_ref}) completed")
    
    # Create heatmap for Wasserstein distances
    wasserstein_dists = np.zeros((len(ds), len(mus)))
    
    for row, mu in enumerate(mus):
        for col, d in enumerate(ds):
            cal = MultiDW(
                N=1000,
                d=d,
                mu=mu,
                t=snapshots[0]+1,
                topology="full",
                num_of_runs=5,
                snapshots=snapshots,
            )
            stats = cal.run()
            std, cluster_count, kde, hist, entropy, observations = stats
            
            # Calculate Wasserstein distance for each run and take the mean
            wasserstein_dists_per_run = []
            for run_obs in observations:
                curr_obs = np.array(run_obs[0])  # Current run, first snapshot
                peaks = len(find_peaks(curr_obs)[0])  # Find peaks
                wasserstein_dists_per_run.append(wasserstein_distance(ref_obs, curr_obs) + np.abs(peaks - peaks_ref))  # Combine distance with peak count difference
            
            wasserstein_dists[row, col] = np.mean(wasserstein_dists_per_run)
            
            print(f"Completed d={d}, mu={mu}")
    
    # Calculate extent to align pixels with data points
    data_step = ds[1] - ds[0]
    extent_min = ds[0] - data_step / 2
    extent_max = ds[-1] + data_step / 2
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(wasserstein_dists, extent=(extent_min, extent_max, extent_min, extent_max), origin="lower", aspect="auto")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{val:.2f}" for val in ds])
    ax.set_yticks(mus)
    ax.set_yticklabels([f"{val:.2f}" for val in mus])
    plt.colorbar(im, label="Wasserstein Distance")
    plt.title(f"Wasserstein Distance (d={d_ref}, mu={mu_ref})")
    plt.xlabel("d")
    plt.ylabel("mu")
    plt.tight_layout()
    plt.savefig(f"wasserstein_heatmap_d{d_ref}_mu{mu_ref}_{snapshot}.png")
    plt.close()

def polarization_heatmap(snapshot):
    """
    Create heatmaps showing the polarization
    between the model with reference parameters (d_ref, mu_ref) and
    each grid point in the d-mu parameter space.
    
    Args:
        d_ref: Reference d parameter
        mu_ref: Reference mu parameter
        snapshot: Time snapshot to analyze
    """
    ds = np.linspace(0.05, 0.5, 10)
    mus = np.linspace(0.05, 0.5, 10)
    snapshots = [snapshot]

    # Create heatmap for polarization
    polarization_values = np.zeros((len(ds), len(mus)))
    
    for row, mu in enumerate(mus):
        for col, d in enumerate(ds):
            cal = MultiDW(
                N=1000,
                d=d,
                mu=mu,
                t=snapshot+1,
                topology="full",
                num_of_runs=5,
                snapshots=snapshots,
            )
            stats = cal.run()
            std, cluster_count, kde, hist, entropy, observations = stats
            
            # Calculate polarization for each run and take the mean
            polarizations_per_run = []
            for run_obs in observations:
                curr_obs = np.array(run_obs[0])  # Current run, first snapshot
                polarizations_per_run.append(_calculate_polarization(curr_obs))
            
            polarization_values[row, col] = np.mean(polarizations_per_run)
            
            print(f"Completed d={d}, mu={mu}")
    
    # Calculate extent to align pixels with data points
    data_step = ds[1] - ds[0]
    extent_min = ds[0] - data_step / 2
    extent_max = ds[-1] + data_step / 2
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(polarization_values, extent=(extent_min, extent_max, extent_min, extent_max), origin="lower", aspect="auto")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{val:.2f}" for val in ds])
    ax.set_yticks(mus)
    ax.set_yticklabels([f"{val:.2f}" for val in mus])
    plt.colorbar(im, label="Polarization (Esteban-Ray)")
    plt.title(f"Polarization t={snapshot+1})")
    plt.xlabel("d")
    plt.ylabel("mu")
    plt.tight_layout()
    plt.savefig(f"polarization_heatmap_{snapshot}.png")
    plt.close()


def _calculate_polarization(observations, alpha=0.0):
    """
    Calculate Esteban-Ray polarization coefficient from agent observations.
    
    Args:
        observations: Array of agent opinion values
        alpha: Freedom degree parameter (default 0.0)
    
    Returns:
        Polarization coefficient
    """
    # Create bins for opinion values
    n_bins = 100 # Adaptive binning
    bins = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(observations, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Create DataFrame with bin distributions and centers
    # pi: proportion of agents in each bin
    # y: opinion value (bin center)
    valid_bins = bin_counts > 0
    pi_values = bin_counts[valid_bins] / len(observations)
    y_values = bin_centers[valid_bins]
    
    data = pd.DataFrame({
        'pi': pi_values,
        'y': y_values
    })
    
    try:
        polarization = esteban_ray(data, pi='pi', y='y', alpha=alpha)
    except Exception as e:
        print(f"Error calculating polarization: {e}")
        polarization = 0.0
    
    return polarization


if __name__ == "__main__":
    # test_entropy_std()
    # kde_difference_heatmap(d_ref=0.1, mu_ref=0.1)
    # kde_difference_heatmap(d_ref=0.4, mu_ref=0.4)
    wasserstein_distance_heatmap(d_ref=0.25, mu_ref=0.25, snapshot=50)
    # polarization_heatmap(snapshot=14)



