import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar_plots_sa_cooling_rate():
    """
    Load SA model data from JSONL files and create bar plots
    for each statistic with cooling rate on x-axis.
    """
    # Load all JSONL files from results directory
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")

    all_data = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("model") == "SA":  # Filter for SA model only
                    all_data.append(data)

    df = pd.DataFrame(all_data)

    # Statistics to plot
    stats = ["prediction_error", "total_time", "abm_calls", "fitness"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        # Group by cooling_rate and calculate mean
        grouped = df.groupby("cooling_rate")[stat].mean().sort_index()
        axes[idx].bar(
            range(len(grouped)), grouped.values, color="steelblue", edgecolor="black"
        )
        axes[idx].set_title(f"SA Model: {stat}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Cooling Rate", fontsize=11)
        axes[idx].set_ylabel(f"Average {stat}", fontsize=11)
        axes[idx].set_xticks(range(len(grouped)))
        axes[idx].set_xticklabels([f"{val:.2f}" for val in grouped.index], fontsize=10)
        axes[idx].grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (x, value) in enumerate(zip(range(len(grouped)), grouped.values)):
            axes[idx].text(
                x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9
            )

    fig.tight_layout()
    fig.savefig("results/sa_cooling_rate_stats.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results/sa_cooling_rate_stats.png")


def heatmaps_ml_models():
    """
    Load data for GBR, RFR, MLP, XGB models and create heatmaps
    for each statistic with pool_size (y-axis) and sample_size (x-axis).
    """
    # Load all JSONL files from results directory
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")

    all_data = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                model = data.get("model")
                if model in ["GBR", "RFR", "MLP", "XGB"]:
                    all_data.append(data)

    df = pd.DataFrame(all_data)
    models = ["GBR", "RFR", "MLP", "XGB"]
    stats = ["prediction_error", "total_time", "abm_calls", "fitness"]

    # Create a figure for each statistic
    for stat in stats:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, model in enumerate(models):
            # Filter data for this model
            model_df = df[df["model"] == model]

            # Create pivot table with pool_size (rows) and sample_size (columns)
            pivot = model_df.pivot_table(
                values=stat, index="pool_size", columns="sample_size", aggfunc="mean"
            )

            # Sort indices to ensure consistent ordering
            pivot = pivot.reindex(sorted(pivot.index)).reindex(
                sorted(pivot.columns), axis=1
            )
            values = pivot.values

            # Create heatmap
            im = axes[idx].imshow(
                values, interpolation="nearest", aspect="auto", cmap="viridis"
            )
            axes[idx].set_title(f"{model}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Sample Size", fontsize=11)
            axes[idx].set_ylabel("Pool Size", fontsize=11)
            axes[idx].set_xticks(np.arange(len(pivot.columns)))
            axes[idx].set_yticks(np.arange(len(pivot.index)))
            axes[idx].set_xticklabels([f"{int(x)}" for x in pivot.columns], fontsize=10)
            axes[idx].set_yticklabels([f"{int(y)}" for y in pivot.index], fontsize=10)

            # Add value labels on heatmap
            for y in range(len(pivot.index)):
                for x in range(len(pivot.columns)):
                    axes[idx].text(
                        x,
                        y,
                        f"{values[y, x]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                    )

            # Add colorbar
            fig.colorbar(im, ax=axes[idx])

        fig.suptitle(f"ML Models: {stat}", fontsize=14, fontweight="bold", y=1.00)
        fig.tight_layout()
        fig.savefig(
            f"results/ml_models_heatmap_{stat}.png", dpi=150, bbox_inches="tight"
        )
        print(f"Plot saved to results/ml_models_heatmap_{stat}.png")
        plt.close()

    
def heatmaps_ga_models():
    """
    Load data for GA1, GA2 models and create heatmaps
    for each statistic with p_c, p_m, pop_size.
    """
    # Load all JSONL files from results directory
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")

    models = ["GA1", "GA2"]

    all_data = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                model = data.get("model")
                if model in models:
                    all_data.append(data)

    df = pd.DataFrame(all_data)
    stats = ["prediction_error", "total_time", "abm_calls", "fitness"]
    pcs = sorted(df["pc"].unique())

    fig, axes = plt.subplots(3, figsize=(14, 10))
    axes = axes.flatten()
    for stat in stats:
        for model in models:
            for idx, pc in enumerate(pcs):

                pc_df: pd.DataFrame = df[df["pcs"] == pc and df["model"] == model]

                pivot = pc_df.pivot_table(
                    values=stat, index="pm", columns="pop_size", aggfunc="mean"
                )

                pivot = pivot.reindex(sorted(pivot.index)).reindex(
                    sorted(pivot.columns), axis=1
                )
                values = pivot.values

                # Create heatmap
                im = axes[idx].imshow(
                    values, interpolation="nearest", aspect="auto", cmap="viridis"
                )
                axes[idx].set_title(f"p_c = {pc}", fontsize=12, fontweight="bold")
                axes[idx].set_xlabel("p_m", fontsize=11)
                axes[idx].set_ylabel("Population Size", fontsize=11)
                axes[idx].set_xticks(np.arange(len(pivot.columns)))
                axes[idx].set_yticks(np.arange(len(pivot.index)))
                axes[idx].set_xticklabels([f"{int(x)}" for x in pivot.columns], fontsize=10)
                axes[idx].set_yticklabels([f"{int(y)}" for y in pivot.index], fontsize=10)

                # Add value labels on heatmap
                for y in range(len(pivot.index)):
                    for x in range(len(pivot.columns)):
                        axes[idx].text(
                            x,
                            y,
                            f"{values[y, x]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="white",
                        )

                # Add colorbar
                fig.colorbar(im, ax=axes[idx])

            fig.suptitle(f"{model} Model: {stat}", fontsize=14, fontweight="bold", y=1.00)
            fig.tight_layout()
            fig.savefig(
                f"results/{model}_heatmap_{stat}.png", dpi=150, bbox_inches="tight"
            )
            print(f"Plot saved to results/{model}_heatmap_{stat}.png")
            plt.close()


if __name__ == "__main__":
    # bar_plots_sa_cooling_rate()
    # heatmaps_ml_models()
    pass
