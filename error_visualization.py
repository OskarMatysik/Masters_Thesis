import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.multiple_runs import MultiDW

def create_error_visualization(d, mu):
    """
    Create visualizations combining entropy heatmap with scatter plot
    of calibration results from a CSV file.
    
    Args:
        d: Confidence threshold parameter
        mu: Convergence rate parameter
    """
    # Generate entropy heatmap data grid
    ds = np.linspace(0.05, 0.5, 10)
    mus = np.linspace(0.05, 0.5, 10)
    
    # Build CSV and JSONL filenames using d and mu parameters
    csv_file = f"results/o_N1000_d{d}_mu{mu}_full.csv"
    jsonl_file = f"results/calibration_results_o_N1000_d{d}_mu{mu}_full.jsonl"
    
    # Read CSV to get snapshots from column names
    csv_df = pd.read_csv(csv_file)
    snapshots = [int(col) for col in csv_df.columns if col not in ['index']]
    
    # Read JSONL file as pandas dataframe
    jsonl_df = pd.read_json(jsonl_file, lines=True)
    d_vals = jsonl_df['d'].tolist()
    mu_vals = jsonl_df['mu'].tolist()
    errors = jsonl_df['prediction_error'].tolist()
    
    print(f"Using snapshots from CSV columns: {snapshots}")
    
    # Create a plot for each snapshot
    for snapshot in snapshots:
        entropies = np.zeros((len(ds), len(mus)))
        
        print(f"\nGenerating entropy heatmap for snapshot {snapshot}...")
        for row, mu_val in enumerate(mus):
            for col, d_val in enumerate(ds):
                cal = MultiDW(
                    N=1000,
                    d=d_val,
                    mu=mu_val,
                    t=snapshot,
                    topology="full",
                    num_of_runs=1,
                )
                stats = cal.run()
                std, cluster, kde, hist, entropy, observations = stats
                entropies[row, col] = entropy[0]
                print(f"Completed d={d_val:.2f}, mu={mu_val:.2f}")
        
        # Create visualization
        print(f"Creating visualization for snapshot {snapshot}...")
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Calculate extent to align pixels with data points
        data_step = ds[1] - ds[0]
        extent_min = ds[0] - data_step / 2
        extent_max = ds[-1] + data_step / 2
        
        # Plot entropy heatmap as background
        im = ax.imshow(
            entropies, 
            extent=(extent_min, extent_max, extent_min, extent_max), 
            origin="lower", 
            aspect="auto",
            cmap="viridis",
            alpha=0.7
        )
        cbar_entropy = plt.colorbar(im, ax=ax, label="Entropy", pad=0.02)
        
        # Overlay scatter plot with prediction error
        scatter = ax.scatter(
            d_vals, 
            mu_vals, 
            c=errors, 
            s=100, 
            cmap="RdYlBu_r",
            edgecolors="black",
            linewidths=1.5,
            alpha=0.8,
            vmin=min(errors),
            vmax=max(errors)
        )
        cbar_error = plt.colorbar(scatter, ax=ax, label="Prediction Error", pad=0.15)
        
        # Set ticks at data point locations
        ax.set_xticks(ds)
        ax.set_xticklabels([f"{val:.2f}" for val in ds])
        ax.set_yticks(mus)
        ax.set_yticklabels([f"{val:.2f}" for val in mus])
        
        ax.set_xlabel("d (Confidence Threshold)", fontsize=12, fontweight="bold")
        ax.set_ylabel("mu (Convergence Rate)", fontsize=12, fontweight="bold")
        ax.set_title(f"Entropy Heatmap with Calibration Results Overlay (Snapshot {snapshot})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        
        fig.tight_layout()
        fig.savefig(f"results/error_visualization_snapshot_{snapshot}.png", dpi=300, bbox_inches="tight")
        print(f"\nVisualization saved to results/error_visualization_snapshot_{snapshot}.png")
        plt.close()


def plot_parameter_space_with_real_values(parameter_sets):
    """
    Create a scatter plot of calibration results across multiple parameter sets
    with a bright green dot marking the real parameter values.
    
    Args:
        parameter_sets: List of tuples [(d1, mu1), (d2, mu2), ...] for which to plot
        d_real: Real confidence threshold parameter
        mu_real: Real convergence rate parameter
    """
    # Collect all data from JSONL files for the given parameter sets
    all_d_vals = []
    all_mu_vals = []
    all_errors = []
    
    for d, mu in parameter_sets:
        jsonl_file = f"results/calibration_results_o_N1000_d{d}_mu{mu}_full.jsonl"
        try:
            jsonl_df = pd.read_json(jsonl_file, lines=True)
            all_d_vals.extend(jsonl_df['d'].tolist())
            all_mu_vals.extend(jsonl_df['mu'].tolist())
            all_errors.extend(jsonl_df['prediction_error'].tolist())
            print(f"Loaded data from d={d}, mu={mu}")
        except FileNotFoundError:
            print(f"File not found: {jsonl_file}")
    
    # Create visualization
    print("\nCreating parameter space scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot calibration results as scatter plot
    scatter = ax.scatter(
        all_d_vals, 
        all_mu_vals, 
        c=all_errors, 
        s=100, 
        cmap="RdYlBu_r",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.8,
        vmin=min(all_errors) if all_errors else 0,
        vmax=max(all_errors) if all_errors else 1,
        label="Calibration Results"
    )
    cbar_error = plt.colorbar(scatter, ax=ax, label="Prediction Error", pad=0.02)
    
    # Add bright green dot for real parameter values
    ax.scatter(
        [d for d, _ in parameter_sets], 
        [mu for _, mu in parameter_sets], 
        c='lime', 
        s=400, 
        edgecolors='darkgreen',
        linewidths=2.5,
        alpha=1.0,
        marker='*',
        label=f"Real Values",
        zorder=5
    )
    
    ax.set_xlabel("d (Confidence Threshold)", fontsize=12, fontweight="bold")
    ax.set_ylabel("mu (Convergence Rate)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_title("Parameter Space with Calibration Results", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc='best')
    
    fig.tight_layout()
    fig.savefig(f"parameter_space_scatter_{parameter_sets[0]}.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # create_error_visualization(0.2, 0.2)
    # plot_parameter_space_with_real_values(parameter_sets=[(0.1, 0.1)])
    # plot_parameter_space_with_real_values(parameter_sets=[(0.2, 0.2)])
    # plot_parameter_space_with_real_values(parameter_sets=[(0.4, 0.4)])
    plot_parameter_space_with_real_values(parameter_sets=[(0.1, 0.4)])

