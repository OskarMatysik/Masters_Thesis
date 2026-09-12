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
            std, cluster, hist, entropy, observations = stats
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
    std, cluster_count, hist_ref, entropy, observations_ref = stats_ref
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
            std, cluster_count, hist, entropy, observations = stats
            
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
            std, cluster_count, hist, entropy, observations = stats
            
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


def all_methods(d_ref, mu_ref, snapshot1, snapshot2):
    """
    Create fitness heatmaps based on differences in multiple statistics across two snapshots.
    Calculates fitness as 1/(1 + difference_at_snapshot1 + difference_at_snapshot2) for:
    - Entropy difference
    - Wasserstein distance difference
    - Polarization difference
    - Histogram sum of absolute differences
    
    Args:
        d_ref: Reference d parameter
        mu_ref: Reference mu parameter
        snapshot1: First time snapshot to analyze
        snapshot2: Second time snapshot to analyze
    """
    ds = np.linspace(0.05, 0.5, 10)
    mus = np.linspace(0.05, 0.5, 10)
    max_snapshot = max(snapshot1, snapshot2)
    snapshots = [snapshot1, snapshot2]
    
    # Get reference statistics for both snapshots
    cal_ref = MultiDW(
        N=1000,
        d=d_ref,
        mu=mu_ref,
        t=max_snapshot+1,
        topology="full",
        num_of_runs=5,
        snapshots=snapshots,
    )
    stats_ref = cal_ref.run()
    std_ref, cluster_count_ref, hist_ref, entropy_ref, observations_ref = stats_ref
    
    # Extract reference values for both snapshots
    ref_entropy_s1 = entropy_ref[0]  # First snapshot
    ref_entropy_s2 = entropy_ref[1]  # Second snapshot
    ref_std_s1 = std_ref[0]
    ref_std_s2 = std_ref[1]
    ref_hist_s1 = np.array(hist_ref[0]) / np.sum(hist_ref[0])  # Normalize to probabilities
    ref_hist_s2 = np.array(hist_ref[1]) / np.sum(hist_ref[1])
    
    # Get reference observations and compute derived metrics for both snapshots
    ref_obs_list_s1 = [np.array(obs[0]) for obs in observations_ref]
    ref_obs_list_s2 = [np.array(obs[1]) for obs in observations_ref]
    
    # Reference polarization for both snapshots
    ref_polarizations_s1 = [_calculate_polarization(obs) for obs in ref_obs_list_s1]
    ref_polarization_s1 = np.mean(ref_polarizations_s1)
    ref_polarizations_s2 = [_calculate_polarization(obs) for obs in ref_obs_list_s2]
    ref_polarization_s2 = np.mean(ref_polarizations_s2)
    
    # Reference Wasserstein for both snapshots
    ref_wasserstein_dists_s1 = [wasserstein_distance(ref_obs_list_s1[0], obs) for obs in ref_obs_list_s1[1:]]
    ref_wasserstein_s1 = np.mean(ref_wasserstein_dists_s1) if ref_wasserstein_dists_s1 else 0.0
    ref_wasserstein_dists_s2 = [wasserstein_distance(ref_obs_list_s2[0], obs) for obs in ref_obs_list_s2[1:]]
    ref_wasserstein_s2 = np.mean(ref_wasserstein_dists_s2) if ref_wasserstein_dists_s2 else 0.0
    
    print(f"Reference model (d={d_ref}, mu={mu_ref}) completed")
    print(f"  Snapshot {snapshot1}: Entropy={ref_entropy_s1:.6f}, Std={ref_std_s1:.6f}, Polarization={ref_polarization_s1:.6f}, Wasserstein={ref_wasserstein_s1:.6f}")
    print(f"  Snapshot {snapshot2}: Entropy={ref_entropy_s2:.6f}, Std={ref_std_s2:.6f}, Polarization={ref_polarization_s2:.6f}, Wasserstein={ref_wasserstein_s2:.6f}")
    
    # Initialize fitness arrays
    entropy_fitness = np.zeros((len(mus), len(ds)))
    std_fitness = np.zeros((len(mus), len(ds)))
    wasserstein_fitness = np.zeros((len(mus), len(ds)))
    polarization_fitness = np.zeros((len(mus), len(ds)))
    histogram_fitness = np.zeros((len(mus), len(ds)))
    combined_fitness = np.zeros((len(mus), len(ds)))

    for row, mu in enumerate(mus):
        for col, d in enumerate(ds):
            cal = MultiDW(
                N=1000,
                d=d,
                mu=mu,
                t=max_snapshot+1,
                topology="full",
                num_of_runs=20,
                snapshots=snapshots,
            )
            stats = cal.run()
            std, cluster_count, hist, entropy, observations = stats
            
            # Extract values for both snapshots
            curr_entropy_s1 = entropy[0]
            curr_entropy_s2 = entropy[1]
            curr_std_s1 = std[0]
            curr_std_s2 = std[1]
            curr_hist_s1 = np.array(hist[0]) / np.sum(hist[0])
            curr_hist_s2 = np.array(hist[1]) / np.sum(hist[1])
            
            # Get observations for both snapshots
            curr_obs_list_s1 = [np.array(obs[0]) for obs in observations]
            curr_obs_list_s2 = [np.array(obs[1]) for obs in observations]
            
            # Calculate polarization for both snapshots
            curr_polarizations_s1 = [_calculate_polarization(obs) for obs in curr_obs_list_s1]
            curr_polarization_s1 = np.mean(curr_polarizations_s1)
            curr_polarizations_s2 = [_calculate_polarization(obs) for obs in curr_obs_list_s2]
            curr_polarization_s2 = np.mean(curr_polarizations_s2)
            
            # Calculate Wasserstein distances for both snapshots
            curr_wasserstein_dists_s1 = [wasserstein_distance(ref_obs_list_s1[0], obs) for obs in curr_obs_list_s1]
            curr_wasserstein_s1 = np.mean(curr_wasserstein_dists_s1)
            curr_wasserstein_dists_s2 = [wasserstein_distance(ref_obs_list_s2[0], obs) for obs in curr_obs_list_s2]
            curr_wasserstein_s2 = np.mean(curr_wasserstein_dists_s2)
            
            # Calculate differences for both snapshots and combine
            entropy_diff_s1 = np.abs(curr_entropy_s1 - ref_entropy_s1)
            entropy_diff_s2 = np.abs(curr_entropy_s2 - ref_entropy_s2)
            entropy_fitness[row, col] = 1.0 / (1.0 + entropy_diff_s1 + entropy_diff_s2)
            
            std_diff_s1 = np.abs(curr_std_s1 - ref_std_s1)
            std_diff_s2 = np.abs(curr_std_s2 - ref_std_s2)
            std_fitness[row, col] = 1.0 / (1.0 + std_diff_s1 + std_diff_s2)
            
            wasserstein_diff_s1 = np.abs(curr_wasserstein_s1 - ref_wasserstein_s1)
            wasserstein_diff_s2 = np.abs(curr_wasserstein_s2 - ref_wasserstein_s2)
            wasserstein_fitness[row, col] = 1.0 / (1.0 + wasserstein_diff_s1 + wasserstein_diff_s2)
            
            polarization_diff_s1 = np.abs(curr_polarization_s1 - ref_polarization_s1)
            polarization_diff_s2 = np.abs(curr_polarization_s2 - ref_polarization_s2)
            polarization_fitness[row, col] = 1.0 / (1.0 + polarization_diff_s1 + polarization_diff_s2)
            
            hist_diff_s1 = np.sum(np.abs(curr_hist_s1 - ref_hist_s1))
            hist_diff_s2 = np.sum(np.abs(curr_hist_s2 - ref_hist_s2))
            histogram_fitness[row, col] = 1.0 / (1.0 + hist_diff_s1 + hist_diff_s2)
            
            # Combined fitness: weighted combination of histogram and Wasserstein distance
            combined_metric = ((hist_diff_s1 + hist_diff_s2) / 4.0) + ((wasserstein_diff_s1 + wasserstein_diff_s2) / 2.0)
            combined_fitness[row, col] = 1.0 / (1.0 + combined_metric)
            
            # Debug on first iteration
            if row == 0 and col == 0:
                print(f"  First point (d={d}, mu={mu}) - Snapshot 1: entropy_diff={entropy_diff_s1:.6f}, Snapshot 2: entropy_diff={entropy_diff_s2:.6f}")
            
            print(f"Completed d={d}, mu={mu}")
    
    # Calculate extent to align pixels with data points
    data_step = ds[1] - ds[0]
    extent_min = ds[0] - data_step / 2
    extent_max = ds[-1] + data_step / 2
    
    # Create heatmaps for each metric
    metrics = [
        ("Entropy", entropy_fitness),
        ("Std Deviation", std_fitness),
        ("Wasserstein Distance", wasserstein_fitness),
        ("Polarization", polarization_fitness),
        ("Histogram", histogram_fitness),
        ("Combined", combined_fitness),
    ]
    
    for metric_name, fitness_values in metrics:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        im = ax.imshow(fitness_values, extent=(extent_min, extent_max, extent_min, extent_max), 
                       origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(ds)
        ax.set_xticklabels([f"{val:.2f}" for val in ds], fontsize=18)
        ax.set_yticks(mus)
        ax.set_yticklabels([f"{val:.2f}" for val in mus], fontsize=18)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel("Fitness", fontsize=16, fontweight="bold")
        plt.title(f"{metric_name} Fitness at t={snapshot1+1},{snapshot2+1} (ref: d={d_ref}, mu={mu_ref})", fontsize=18, fontweight="bold")
        plt.xlabel("d", fontsize=16)
        plt.ylabel("mu", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"fitness_{metric_name.lower().replace(' ', '_')}_d{d_ref}_mu{mu_ref}_t{snapshot1+1}_{snapshot2+1}.png")
        plt.close()
        print(f"Saved fitness_{metric_name.lower().replace(' ', '_')}_d{d_ref}_mu{mu_ref}_t{snapshot1+1}_{snapshot2+1}.png")


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
    # wasserstein_distance_heatmap(d_ref=0.25, mu_ref=0.25, snapshot=50)
    # polarization_heatmap(snapshot=14)
    # all_methods(d_ref=0.25, mu_ref=0.25, snapshot=14)
    all_methods(d_ref=0.1, mu_ref=0.4, snapshot1=14, snapshot2=49)
    all_methods(d_ref=0.4, mu_ref=0.1, snapshot1=14, snapshot2=49)
    all_methods(d_ref=0.25, mu_ref=0.25, snapshot1=14, snapshot2=49)



