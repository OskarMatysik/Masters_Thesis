import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heatmaps_ml(filenames, ml_models, stat_name):

    fig = plt.figure(figsize=(16, 48))
    gs = fig.add_gridspec(9, 3, width_ratios=[1, 1, 1], wspace=0.3)

    ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(9)])

    for i, dataset_name in enumerate(filenames):
        for j, model in enumerate(ml_models):
            file_name = f"results/results_{model}{dataset_name}"
            df = pd.read_csv(file_name)[["pool_size", "sample_size", stat_name]]
            values = df[stat_name].values.reshape(3, 3)

            pool_sizes = sorted(df["pool_size"].unique())
            sample_sizes = sorted(df["sample_size"].unique())

            ax[i, j].imshow(values, interpolation="nearest")
            ax[i, j].set_title(f"{model} - {dataset_name}", fontsize=12)
            ax[i, j].set_xlabel("Sample Size", fontsize=11)
            ax[i, j].set_ylabel("Pool Size", fontsize=11)
            ax[i, j].set_xticks(np.arange(3))
            ax[i, j].set_yticks(np.arange(3))
            ax[i, j].set_xticklabels(sample_sizes, fontsize=10)
            ax[i, j].set_yticklabels(pool_sizes, fontsize=10)

            for y in range(3):
                for x in range(3):
                    ax[i, j].text(
                        x,
                        y,
                        f"{values[y, x]:.4f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )

    plt.savefig(f"results/heatmaps_ml_{stat_name}.png", bbox_inches="tight")


def heatmaps_avg_ml(filenames, ml_models, stat_name):

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    pool_sizes = None
    sample_sizes = None

    for j, model in enumerate(ml_models):
        values = np.zeros((3, 3))

        for i, dataset_name in enumerate(filenames):
            file_name = f"results/results_{model}{dataset_name}"
            df = pd.read_csv(file_name)[["pool_size", "sample_size", stat_name]]
            values += df[stat_name].values.reshape(3, 3)

            # Get pool_sizes and sample_sizes from the first iteration
            if pool_sizes is None:
                pool_sizes = sorted(df["pool_size"].unique())
                sample_sizes = sorted(df["sample_size"].unique())

        values /= len(filenames)

        ax[j].imshow(values, interpolation="nearest")
        ax[j].set_title(f"{model}", fontsize=12)
        ax[j].set_xlabel("Sample Size", fontsize=11)
        ax[j].set_ylabel("Pool Size", fontsize=11)
        ax[j].set_xticks(np.arange(3))
        ax[j].set_yticks(np.arange(3))
        ax[j].set_xticklabels(sample_sizes, fontsize=10)
        ax[j].set_yticklabels(pool_sizes, fontsize=10)

        for y in range(3):
            for x in range(3):
                ax[j].text(
                    x, y, f"{values[y, x]:.4f}", ha="center", va="center", fontsize=10
                )

    plt.savefig(f"results/heatmaps_avg_ml_{stat_name}.png", bbox_inches="tight")


def heatmaps_avg_GA1(file_names, stat_name):
    """
    Create bar plots showing the relationship between GA1 parameters
    (pc, pm, pop_size) and a statistic, averaged across all datasets.
    """
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    parameters = ["pc", "pm", "pop_size"]

    # Aggregate data across all datasets
    all_data = []

    for dataset_name in file_names:
        file_name = f"results/results_GA1{dataset_name}"
        df = pd.read_csv(file_name)[["pc", "pm", "pop_size", stat_name]]
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Create bar plots for each parameter
    for idx, param in enumerate(parameters):
        # Group by parameter and calculate mean of stat_name
        grouped = combined_df.groupby(param)[stat_name].mean().sort_index()

        bars = ax[idx].bar(range(len(grouped)), grouped.values, color="steelblue")
        ax[idx].set_title(f"{param}", fontsize=12)
        ax[idx].set_xlabel(f"{param}", fontsize=11)
        ax[idx].set_ylabel(f"Average {stat_name}", fontsize=11)
        ax[idx].set_xticks(range(len(grouped)))
        ax[idx].set_xticklabels(
            [
                f"{val:.4g}" if isinstance(val, float) else str(val)
                for val in grouped.index
            ],
            fontsize=10,
        )

        # Add value labels on bars
        for bar, value in zip(bars, grouped.values):
            ax[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.savefig(f"results/heatmaps_avg_GA1_{stat_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    # Dataset parameters
    topologies = ["full"]
    Ns = [1000]
    ds = [0.1, 0.25, 0.4]
    mus = [0.1, 0.25, 0.4]

    stats_names = ["prediction_error", "total_time", "abm_calls"]

    # ML
    ml_models = ["GBR", "RFR", "MLP"]
    ml_file_names = [
        f"_o_N{N}_d{d}_mu{mu}_{topology}.csv"
        for topology in topologies
        for N in Ns
        for d in ds
        for mu in mus
    ]

    # GA1
    ga1_file_names = [
        f"_o_N{N}_d{d}_mu{mu}_{topology}.csv"
        for topology in topologies
        for N in Ns
        for d in ds
        for mu in mus
    ]

    for stat_name in stats_names:
        heatmaps_avg_ml(ml_file_names, ml_models, stat_name)
        heatmaps_avg_GA1(ga1_file_names, stat_name)
