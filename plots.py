import glob
import json

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import re

def bar_plots_sa_cooling_rate():
    """
    Load SA model data from JSONL files and create heatmaps
    for each statistic with max_iter (y-axis) and cooling_rate (x-axis).
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

    # Create a heatmap for each statistic
    for stat in stats:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create pivot table with cooling_rate (rows) and max_iter (columns)
        pivot = df.pivot_table(
            values=stat, index="cooling_rate", columns="max_iter", aggfunc="mean"
        )

        # Sort indices to ensure consistent ordering
        pivot = pivot.reindex(sorted(pivot.index)).reindex(
            sorted(pivot.columns), axis=1
        )
        values = pivot.values

        # Create heatmap
        im = ax.imshow(
            values, interpolation="nearest", aspect="auto", cmap="viridis"
        )
        ax.set_title(f"SA Model: {stat}", fontsize=18, fontweight="bold")
        ax.set_xlabel("Max Iterations", fontsize=15)
        ax.set_ylabel("Cooling Rate", fontsize=15)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"{int(x)}" for x in pivot.columns], fontsize=12)
        ax.set_yticklabels([f"{val:.2f}" for val in pivot.index], fontsize=12)

        # Add value labels on heatmap
        for y in range(len(pivot.index)):
            for x in range(len(pivot.columns)):
                value_str = f"{int(values[y, x])}" if stat == "abm_calls" else f"{values[y, x]:.2f}"
                txt = ax.text(
                    x,
                    y,
                    value_str,
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="white",
                )
                txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

        # Add colorbar
        fig.colorbar(im, ax=ax)

        fig.tight_layout()
        fig.savefig(f"sa_cooling_rate_heatmap_{stat}.png", dpi=150, bbox_inches="tight")
        print(f"Plot saved to sa_cooling_rate_heatmap_{stat}.png")
        plt.close()


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

            # Create pivot table with sample_size (rows) and pool_size (columns)
            pivot = model_df.pivot_table(
                values=stat, index="sample_size", columns="pool_size", aggfunc="mean"
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
            axes[idx].set_title(f"{model}", fontsize=18, fontweight="bold")
            axes[idx].set_xlabel("Pool Size", fontsize=15)
            axes[idx].set_ylabel("Sample Size", fontsize=15)
            axes[idx].set_xticks(np.arange(len(pivot.columns)))
            axes[idx].set_yticks(np.arange(len(pivot.index)))
            axes[idx].set_xticklabels([f"{int(x)}" for x in pivot.columns], fontsize=12)
            axes[idx].set_yticklabels([f"{int(y)}" for y in pivot.index], fontsize=12)

            # Add value labels on heatmap
            for y in range(len(pivot.index)):
                for x in range(len(pivot.columns)):
                    value_str = f"{int(values[y, x])}" if stat == "abm_calls" else f"{values[y, x]:.2f}"
                    txt = axes[idx].text(
                        x,
                        y,
                        value_str,
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="white",
                    )
                    txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

            # Add colorbar
            fig.colorbar(im, ax=axes[idx])

        fig.suptitle(f"ML Models: {stat}", fontsize=18, fontweight="bold", y=1.00)
        fig.tight_layout()
        fig.savefig(
            f"ml_models_heatmap_{stat}.png", dpi=150, bbox_inches="tight"
        )
        print(f"Plot saved to ml_models_heatmap_{stat}.png")
        plt.close()

    
def heatmaps_ga_models():
    """
    Load data for GA1, GA2 models and create heatmaps
    for each statistic with p_c, p_m, pop_size.
    """
    # Load all JSONL files from results directory
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")

    models = ["GA1"]

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

    for stat in stats:
        for model in models:
            num_pcs = len(pcs)
            fig, axes = plt.subplots(num_pcs, 1, figsize=(5, 4 * num_pcs))
            if num_pcs == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, pc in enumerate(pcs):
                pc_df: pd.DataFrame = df[(df["pc"] == pc) & (df["model"] == model)]

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
                axes[idx].set_title(f"p_c = {pc}", fontsize=18, fontweight="bold")
                axes[idx].set_xlabel("pop_size", fontsize=15)
                axes[idx].set_ylabel("p_m", fontsize=15)
                axes[idx].set_xticks(np.arange(len(pivot.columns)))
                axes[idx].set_yticks(np.arange(len(pivot.index)))
                axes[idx].set_xticklabels([f"{int(x)}" for x in pivot.columns], fontsize=12)
                axes[idx].set_yticklabels([f"{y:.2f}" for y in pivot.index], fontsize=12)

                # Add value labels on heatmap
                for y in range(len(pivot.index)):
                    for x in range(len(pivot.columns)):
                        value_str = f"{int(values[y, x])}" if stat == "abm_calls" else f"{values[y, x]:.2f}"
                        txt = axes[idx].text(
                            x,
                            y,
                            value_str,
                            ha="center",
                            va="center",
                            fontsize=14,
                            color="white",
                        )
                        txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

                # Add colorbar
                fig.colorbar(im, ax=axes[idx])

            fig.suptitle(f"{model} Model: {stat}", fontsize=18, fontweight="bold", y=1.00)
            fig.tight_layout()
            fig.savefig(
                f"{model}_heatmap_{stat}.png", dpi=150, bbox_inches="tight"
            )
            print(f"Plot saved to {model}_heatmap_{stat}.png")
            plt.close()


def bar_plot_grid_search():
    """
    Create four bar plots for grid search statistics where x-axis is grid_size.
    Each plot represents one statistic: prediction_error, abm_calls, total_time, fitness.
    """
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")

    all_data = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                model = data.get("model")
                if model == "GS":
                    all_data.append(data)

    df = pd.DataFrame(all_data)

    # Hardcode correct ABM calls for grid search and scale time accordingly
    correct_abm_calls = {23: 10580, 32: 20480, 45: 40500}
    
    for grid_size, correct_abm in correct_abm_calls.items():
        mask = df["grid_size"] == grid_size
        if mask.any():
            actual_abm = df.loc[mask, "abm_calls"].mean()
            scaling_factor = correct_abm / actual_abm if actual_abm > 0 else 1.0
            # Scale time proportionally to ABM calls
            df.loc[mask, "abm_calls"] = correct_abm
            df.loc[mask, "total_time"] = df.loc[mask, "total_time"] * scaling_factor

    # Statistics to plot
    stats = ["prediction_error", "abm_calls", "total_time", "fitness"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        # Group by grid_size and calculate mean
        grouped = df.groupby("grid_size")[stat].mean().sort_index()
        axes[idx].bar(
            range(len(grouped)), grouped.values, color="steelblue", edgecolor="black"
        )
        axes[idx].set_title(f"Grid Search: {stat}", fontsize=18, fontweight="bold")
        axes[idx].set_xlabel("Grid Size", fontsize=15)
        axes[idx].set_ylabel(f"Average {stat}", fontsize=15)
        axes[idx].set_xticks(range(len(grouped)))
        axes[idx].set_xticklabels([f"{int(val)}" for val in grouped.index], fontsize=12)
        axes[idx].grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (x, value) in enumerate(zip(range(len(grouped)), grouped.values)):
            value_str = f"{int(value)}" if stat == "abm_calls" else f"{value:.2f}"
            axes[idx].text(
                x, value, value_str, ha="center", va="bottom", fontsize=14
            )

    fig.tight_layout()
    fig.savefig("grid_search_stats.png", dpi=150, bbox_inches="tight")
    print("Plot saved to grid_search_stats.png")


def scatter_plot_predictions_vs_real(model, optimal_params=None):
    """
    Create scatter plots comparing real vs predicted d and mu values.
    
    The real d and mu values are extracted from JSONL filenames.
    The d and mu values are read from the JSONL file entries.
    
    Args:
        model (str): The model name (e.g., 'GBR', 'SA', 'GA1')
        optimal_params (dict, optional): Dictionary of parameters to filter by. 
                                         If provided, only results matching these parameters are plotted.
    
    Example:
        scatter_plot_predictions_vs_real('GBR')
        scatter_plot_predictions_vs_real('SA', {'cooling_rate': 0.85, 'max_iter': 512})
    """
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")
    
    all_data = []
    
    for file_path in jsonl_files:
        # Extract real d and mu from filename (e.g., calibration_results_o_N1000_d0.2_mu0.2_full.jsonl)
        match = re.search(r'd([\d.]+)_mu([\d.]+)', file_path)
        if not match:
            continue
        
        real_d = float(match.group(1))
        real_mu = float(match.group(2))
        
        # Read data from JSONL file
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("model") == model:
                    # Filter by optimal parameters if provided
                    if optimal_params:
                        skip = False
                        for param_name, param_value in optimal_params.items():
                            if data.get(param_name) != param_value:
                                skip = True
                                break
                        if skip:
                            continue
                    
                    # Store real values separately from predicted values
                    data["real_d"] = real_d
                    data["real_mu"] = real_mu
                    all_data.append(data)
    
    if not all_data:
        print(f"No data found for model '{model}'")
        return
    
    df = pd.DataFrame(all_data)
    
    # Calculate differences (real - predicted)
    df["d_diff"] = df["real_d"] - df["d"]
    df["mu_diff"] = df["real_mu"] - df["mu"]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # First plot: colored by real_d
    scatter1 = axes[0].scatter(
        df["d_diff"],
        df["mu_diff"],
        alpha=0.6,
        s=20,
        c=df["real_d"],
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5
    )
    
    axes[0].set_xlabel("d Difference (Real - Predicted)", fontsize=15)
    axes[0].set_ylabel("μ Difference (Real - Predicted)", fontsize=15)
    axes[0].set_title(f"{model} Model: Colored by Real d", fontsize=18, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, 0.5)
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    
    cbar1 = fig.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Real d", fontsize=11)
    
    # Second plot: colored by real_mu
    scatter2 = axes[1].scatter(
        df["d_diff"],
        df["mu_diff"],
        alpha=0.6,
        s=20,
        c=df["real_mu"],
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5
    )
    
    axes[1].set_xlabel("d Difference (Real - Predicted)", fontsize=15)
    axes[1].set_ylabel("μ Difference (Real - Predicted)", fontsize=15)
    axes[1].set_title(f"{model} Model: Colored by Real μ", fontsize=18, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.5, 0.5)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    
    cbar2 = fig.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("Real μ", fontsize=11)
    
    fig.tight_layout()
    # Create filename with parameter information if provided
    if optimal_params:
        filename = f"{model}_scatter_predictions_vs_real_optimal.png"
    else:
        filename = f"{model}_scatter_predictions_vs_real.png"
    
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filename}")
    plt.close()


def bar_plot_optimal_parameters(optimal_params):
    """
    Create four bar plots for each statistic, with bars representing
    different methods using their optimal parameters.
    
    Args:
        optimal_params: Dictionary with method names as keys and parameter dicts as values
    """
    # Load all JSONL files from results directory
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")

    all_data = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)

    df = pd.DataFrame(all_data)

    # Filter data for optimal parameters
    optimal_results = {}

    for method, params in optimal_params.items():
        # Start with full dataframe
        mask = df["model"] == method

        # Apply parameter filters
        for param_name, param_value in params.items():
            mask = mask & (df[param_name] == param_value)

        # Get the filtered data
        filtered_df = df[mask]

        if not filtered_df.empty:
            optimal_results[method] = {
                "prediction_error": filtered_df["prediction_error"].mean(),
                "total_time": filtered_df["total_time"].mean(),
                "abm_calls": filtered_df["abm_calls"].mean(),
                "fitness": filtered_df["fitness"].mean(),
            }
        else:
            print(f"No data found for {method} with optimal parameters: {params}")

    # Create bar plots
    stats = ["prediction_error", "abm_calls", "total_time", "fitness"]
    methods = list(optimal_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        values = [optimal_results[method][stat] for method in methods]
        
        # Sort methods and values in descending order
        sorted_pairs = sorted(zip(methods, values), key=lambda x: x[1], reverse=True)
        sorted_methods = [pair[0] for pair in sorted_pairs]
        sorted_values = [pair[1] for pair in sorted_pairs]
        
        # Get colors for sorted methods
        sorted_colors = [colors[methods.index(m)] for m in sorted_methods]

        axes[idx].bar(
            range(len(sorted_methods)), sorted_values, color=sorted_colors, edgecolor="black", linewidth=1.5
        )
        axes[idx].set_title(f"{stat}", fontsize=18, fontweight="bold")
        axes[idx].set_xlabel("Method", fontsize=15)
        axes[idx].set_ylabel(f"Average {stat}", fontsize=15)
        axes[idx].set_xticks(range(len(sorted_methods)))
        axes[idx].set_xticklabels(sorted_methods, fontsize=12, rotation=45)
        axes[idx].grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (method, value) in enumerate(zip(sorted_methods, sorted_values)):
            value_str = (
                f"{int(value)}" if stat == "abm_calls" else f"{value:.2f}"
            )
            axes[idx].text(
                i, value, value_str, ha="center", va="bottom", fontsize=14
            )

    fig.tight_layout()
    fig.savefig("optimal_parameters_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to optimal_parameters_comparison.png")
    plt.close()


def plot_parameter_sensitivity(param_name, param_value, optimal_params=None):
    """
    Create four subplots showing how statistics change across methods for varying d or mu parameter.
    
    Fixes one of the d/mu parameters to a specific value and iterates over the other.
    For each varying parameter value, plots statistics for all methods.
    
    Args:
        param_name (str): Either "d" or "mu" - the parameter to fix
        param_value (float): The value to fix (e.g., 0.2 for d=0.2)
        optimal_params (dict, optional): Dictionary with method names as keys and parameter dicts as values.
                                         If provided, only results matching these parameters are used.
                                         If not provided, averages across all results for each method.
    
    Example:
        plot_parameter_sensitivity("d", 0.2, optimal_params=optimal_params)
        plot_parameter_sensitivity("mu", 0.1)
    """
    # Determine the varying parameter
    if param_name == "d":
        varying_param = "mu"
    elif param_name == "mu":
        varying_param = "d"
    else:
        print(f"param_name must be 'd' or 'mu', got {param_name}")
        return
    
    # Load all JSONL files and extract d/mu from filenames
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")
    all_data = []
    
    for file_path in jsonl_files:
        # Extract d and mu from filename (e.g., calibration_results_o_N1000_d0.2_mu0.2_full.jsonl)
        match = re.search(rf'{param_name}([\d.]+)_{varying_param}([\d.]+)', file_path)
        if not match:
            continue
        
        fixed_val = float(match.group(1))
        varying_val = float(match.group(2))
        
        # Only process files that match the fixed parameter value
        if abs(fixed_val - param_value) > 1e-6:  # Use small epsilon for float comparison
            continue
        
        # Read data from JSONL file
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                data[param_name] = fixed_val
                data[varying_param] = varying_val
                all_data.append(data)
    
    if not all_data:
        print(f"No data found for {param_name} = {param_value}")
        return
    
    df = pd.DataFrame(all_data)
    
    # Get unique methods and varying parameter values
    methods = sorted(df["model"].unique())
    varying_values = sorted(df[varying_param].unique())
    
    # Create 4 subplots for each statistic
    stats = ["prediction_error", "abm_calls", "total_time", "fitness"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for stat_idx, stat in enumerate(stats):
        ax = axes[stat_idx]
        
        # For each method, plot how the stat changes with the varying parameter
        for method_idx, method in enumerate(methods):
            method_data = df[df["model"] == method].copy()
            
            # If optimal_params provided, further filter to those parameters
            if optimal_params and method in optimal_params:
                for opt_param_name, opt_param_value in optimal_params[method].items():
                    if opt_param_name not in [param_name, varying_param]:  # Only apply other parameters
                        method_data = method_data[method_data[opt_param_name] == opt_param_value]
            
            if method_data.empty:
                continue
            
            # Group by varying parameter and calculate mean statistic
            grouped = method_data.groupby(varying_param)[stat].mean().sort_index()
            
            # Plot line for this method
            ax.plot(grouped.index, grouped.values, marker='o', linewidth=2.5, 
                   markersize=8, label=method, color=colors[method_idx])
        
        # Format the subplot
        ax.set_title(stat, fontsize=18, fontweight="bold")
        ax.set_ylabel(f"Average {stat}", fontsize=15)
        ax.set_xlabel(varying_param, fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
    
    fig.suptitle(f"Parameter Sensitivity: {param_name} = {param_value}", fontsize=18, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(f"{param_name}_{param_value}_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {param_name}_{param_value}_sensitivity.png")
    plt.close()


def plot_fitness_threshold_analysis(stop_fitness, optimal_params=None):
    """
    Analyze results grouped by a fitness threshold.
    
    Creates visualizations showing how methods perform relative to a fitness threshold.
    Results below the threshold are marked in red, above in green.
    
    Args:
        stop_fitness (float): The fitness threshold to group results by
        optimal_params (dict, optional): Dictionary with method names as keys and parameter dicts as values.
                                         If provided, only results matching these parameters are used.
                                         If not provided, averages across all results for each method.
    
    Example:
        plot_fitness_threshold_analysis(0.1, optimal_params=optimal_params)
    """
    # Load all JSONL files and extract d/mu from filenames
    jsonl_files = glob.glob("results/calibration_results_*.jsonl")
    all_data = []
    
    for file_path in jsonl_files:
        # Extract d and mu from filename
        match = re.search(r'd([\d.]+)_mu([\d.]+)', file_path)
        if not match:
            continue
        
        real_d = float(match.group(1))
        real_mu = float(match.group(2))
        
        # Read data from JSONL file
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                data["real_d"] = real_d
                data["real_mu"] = real_mu
                all_data.append(data)
    
    df = pd.DataFrame(all_data)
    
    # Classify by fitness threshold
    df["above_threshold"] = df["fitness"] >= stop_fitness
    
    # Get data above threshold
    df_above = df[df["above_threshold"]].copy()
    
    # Apply optimal params filtering if provided
    if optimal_params:
        # Filter each method by its optimal parameters
        filtered_dfs = []
        for method, params in optimal_params.items():
            method_df = df_above[df_above["model"] == method].copy()
            for param_name, param_value in params.items():
                method_df = method_df[method_df[param_name] == param_value]
            if not method_df.empty:
                filtered_dfs.append(method_df)
        df_above_filtered = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else df_above.iloc[0:0]
    else:
        df_above_filtered = df_above
    
    # Calculate averages for each method
    methods = sorted(df_above_filtered["model"].unique())
    abm_calls_by_method = []
    total_time_by_method = []
    prediction_error_by_method = []
    fitness_by_method = []
    
    for method in methods:
        method_data = df_above_filtered[df_above_filtered["model"] == method]
        if not method_data.empty:
            abm_calls_by_method.append(method_data["abm_calls"].mean())
            total_time_by_method.append(method_data["total_time"].mean())
            prediction_error_by_method.append(method_data["prediction_error"].mean())
            fitness_by_method.append(method_data["fitness"].mean())
        else:
            abm_calls_by_method.append(0)
            total_time_by_method.append(0)
            prediction_error_by_method.append(0)
            fitness_by_method.append(0)
    
    # If no optimal_params provided, hardcode GS abm_calls and scale time
    if not optimal_params and "GS" in methods:
        gs_idx = methods.index("GS")
        actual_abm = abm_calls_by_method[gs_idx]
        if actual_abm > 0:
            scaling_factor = 23853 / actual_abm
            abm_calls_by_method[gs_idx] = 23853
            total_time_by_method[gs_idx] = total_time_by_method[gs_idx] * scaling_factor
    
    if not methods or not abm_calls_by_method:
        print(f"No data found above fitness threshold {stop_fitness}")
        return
    
    # Create four bar plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    stats_data = [
        ("abm_calls", abm_calls_by_method, "ABM Calls"),
        ("total_time", total_time_by_method, "Total Time"),
        ("prediction_error", prediction_error_by_method, "Prediction Error"),
        ("fitness", fitness_by_method, "Fitness")
    ]
    
    for plot_idx, (stat_key, values, stat_label) in enumerate(stats_data):
        # Sort by values (descending)
        sorted_pairs = sorted(zip(methods, values), key=lambda x: x[1], reverse=True)
        sorted_methods = [pair[0] for pair in sorted_pairs]
        sorted_values = [pair[1] for pair in sorted_pairs]
        sorted_colors = [colors[methods.index(m)] for m in sorted_methods]
        
        # Plot
        axes[plot_idx].bar(range(len(sorted_methods)), sorted_values, color=sorted_colors, edgecolor="black", linewidth=1.5)
        axes[plot_idx].set_title(f"{stat_label} (fitness >= {stop_fitness})", fontsize=18, fontweight="bold")
        axes[plot_idx].set_xlabel("Method", fontsize=15)
        axes[plot_idx].set_ylabel(f"Average {stat_label}", fontsize=15)
        axes[plot_idx].set_xticks(range(len(sorted_methods)))
        axes[plot_idx].set_xticklabels(sorted_methods, fontsize=12, rotation=45)
        axes[plot_idx].grid(axis="y", alpha=0.3)
        
        # Add value labels
        for i, value in enumerate(sorted_values):
            if stat_key == "abm_calls":
                value_str = f"{int(value)}"
            else:
                value_str = f"{value:.2f}"
            axes[plot_idx].text(i, value, value_str, ha="center", va="bottom", fontsize=12)
    
    fig.tight_layout()
    fig.savefig(f"fitness_threshold_{"optimal" if optimal_params else ""}_stats.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to fitness_threshold_{"optimal" if optimal_params else ""}_stats.png")
    plt.close()
    
    # Create heatmap showing number of models that exceeded stop_fitness per grid point
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Get unique d and mu values from real parameters
    unique_d = sorted(df["real_d"].unique())
    unique_mu = sorted(df["real_mu"].unique())
    
    # Create a grid to count successful models per (d, mu) point
    grid = np.zeros((len(unique_mu), len(unique_d)))
    
    for d_idx, d_val in enumerate(unique_d):
        for mu_idx, mu_val in enumerate(unique_mu):
            # Filter by real d and mu and fitness threshold
            grid_data = df[(df["real_d"] == d_val) & 
                           (df["real_mu"] == mu_val) & 
                           (df["fitness"] >= stop_fitness)]
            
            # If optimal_params provided, further filter by method and parameters
            if optimal_params:
                filtered_dfs = []
                for method, params in optimal_params.items():
                    method_data = grid_data[grid_data["model"] == method].copy()
                    for param_name, param_value in params.items():
                        method_data = method_data[method_data[param_name] == param_value]
                    filtered_dfs.append(method_data)
                count = len(pd.concat(filtered_dfs, ignore_index=True)) if filtered_dfs else 0
            else:
                count = len(grid_data)
            
            grid[mu_idx, d_idx] = count
    
    # Create heatmap
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', origin='lower')
    ax.set_xticks(np.arange(len(unique_d)))
    ax.set_yticks(np.arange(len(unique_mu)))
    ax.set_xticklabels([f"{d:.2f}" for d in unique_d], fontsize=11)
    ax.set_yticklabels([f"{mu:.2f}" for mu in unique_mu], fontsize=11)
    ax.set_xlabel("Real d", fontsize=15, fontweight="bold")
    ax.set_ylabel("Real μ", fontsize=15, fontweight="bold")
    ax.set_title(f"Number of Models Exceeding Fitness {stop_fitness}", fontsize=18, fontweight="bold")
    
    # Add value labels on heatmap
    for mu_idx in range(len(unique_mu)):
        for d_idx in range(len(unique_d)):
            value = int(grid[mu_idx, d_idx])
            txt = ax.text(d_idx, mu_idx, str(value),
                         ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Models", fontsize=12)
    
    fig.tight_layout()
    fig.savefig(f"fitness_threshold_{"optimal" if optimal_params else ""}_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to fitness_threshold_{"optimal" if optimal_params else ""}_heatmap.png")
    plt.close()


def plot_parameter_space_with_real_values(parameter_sets, optimal_params=None):
    """
    Create a scatter plot of calibration results across multiple parameter sets
    with a bright green dot marking the real parameter values.
    
    Args:
        parameter_sets: List of tuples [(d1, mu1), (d2, mu2), ...] for which to plot
        optimal_params: Optional dict where keys are method names and values are parameter dicts.
                       Only calibrations done with these exact parameters will be plotted.
                       Example: {"GBR": {"sample_size": 20, "pool_size": 512}, "SA": {"cooling_rate": 0.85, "max_iter": 512}}
    """
    # Collect all data from JSONL files for the given parameter sets
    all_d_vals = []
    all_mu_vals = []
    all_errors = []
    
    for d, mu in parameter_sets:
        jsonl_file = f"results/calibration_results_o_N1000_d{d}_mu{mu}_full.jsonl"
        try:
            jsonl_df = pd.read_json(jsonl_file, lines=True)
            
            # If optimal parameters are provided, filter the data
            if optimal_params:
                # Filter by method name keys and their specific parameters
                filtered_dfs = []
                for method, params in optimal_params.items():
                    method_df = jsonl_df[jsonl_df['model'] == method].copy()
                    
                    # Apply method-specific parameter filters
                    for param_name, param_value in params.items():
                        if param_name in method_df.columns:
                            method_df = method_df[method_df[param_name] == param_value]
                    
                    filtered_dfs.append(method_df)
                
                jsonl_df = pd.concat(filtered_dfs, ignore_index=True)
                print(f"Loaded {len(jsonl_df)} matching records from d={d}, mu={mu} (filtered by optimal params)")
            else:
                print(f"Loaded data from d={d}, mu={mu}")
            
            all_d_vals.extend(jsonl_df['d'].tolist())
            all_mu_vals.extend(jsonl_df['mu'].tolist())
            all_errors.extend(jsonl_df['prediction_error'].tolist())
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
        label=f"Real Value",
        zorder=5
    )
    
    ax.set_xlabel("d (Confidence Threshold)", fontsize=15, fontweight="bold")
    ax.set_ylabel("mu (Convergence Rate)", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    
    # Add note about filtering in title if optimal params were provided
    title_suffix = " (Optimal Parameters)" if optimal_params else ""
    ax.set_title(f"Parameter Space with Calibration Results{title_suffix}", fontsize=18, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=14, loc='best')
    
    fig.tight_layout()
    
    # Include optimal params info in filename if provided
    filename_suffix = ""
    if optimal_params:
        filename_suffix = "_optimal"
    
    fig.savefig(f"parameter_space_scatter_{parameter_sets[0]}{filename_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    methods = ["GBR", "RFR", "MLP", "XGB", "SA", "GA1", "GS"]
    optimal_params = {
        "GS": {"grid_size": 23},
        "SA": {"cooling_rate": 0.85, "max_iter": 512},
        "GA1": {"pc": 0.8, "pm": 0.1, "pop_size": 8},
        "GBR": {"sample_size": 20, "pool_size": 512},
        "RFR": {"sample_size": 20, "pool_size": 512},
        "MLP": {"sample_size": 20, "pool_size": 512},
        "XGB": {"sample_size": 20, "pool_size": 512},
    }

    # bar_plots_sa_cooling_rate()
    # heatmaps_ml_models()
    # heatmaps_ga_models()
    # bar_plot_grid_search()
    # bar_plot_optimal_parameters(optimal_params)
    # for method in methods:
    #     scatter_plot_predictions_vs_real(method)
    #     scatter_plot_predictions_vs_real(method, optimal_params=optimal_params.get(method))
    # plot_parameter_sensitivity("d", 0.05, optimal_params=optimal_params)
    # plot_parameter_sensitivity("d", 0.25, optimal_params=optimal_params)
    # plot_parameter_sensitivity("d", 0.5, optimal_params=optimal_params)
    # plot_parameter_sensitivity("mu", 0.05, optimal_params=optimal_params)
    # plot_parameter_sensitivity("mu", 0.25, optimal_params=optimal_params)
    # plot_parameter_sensitivity("mu", 0.5, optimal_params=optimal_params)
    # plot_fitness_threshold_analysis(0.9, optimal_params=optimal_params)
    # plot_fitness_threshold_analysis(0.9)

    plot_parameter_space_with_real_values(parameter_sets=[(0.4, 0.1)])
    plot_parameter_space_with_real_values(parameter_sets=[(0.25, 0.25)])
    plot_parameter_space_with_real_values(parameter_sets=[(0.1, 0.4)])
    plot_parameter_space_with_real_values(parameter_sets=[(0.1, 0.1)])
    plot_parameter_space_with_real_values(parameter_sets=[(0.4, 0.4)])
    plot_parameter_space_with_real_values(parameter_sets=[(0.4, 0.1)], optimal_params=optimal_params)
    plot_parameter_space_with_real_values(parameter_sets=[(0.25, 0.25)], optimal_params=optimal_params)
    plot_parameter_space_with_real_values(parameter_sets=[(0.1, 0.4)], optimal_params=optimal_params)
    plot_parameter_space_with_real_values(parameter_sets=[(0.1, 0.1)], optimal_params=optimal_params)
    plot_parameter_space_with_real_values(parameter_sets=[(0.4, 0.4)], optimal_params=optimal_params)


