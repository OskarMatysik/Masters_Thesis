from argparse import ArgumentParser

from tasks import (
    task_calibration_GA1,
    task_calibration_GA2,
    task_calibration_GS,
    task_calibration_ML_surrogate,
    task_calibration_SA,
)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run the main script for dataset generation and calibration tasks."
    )
    parser.add_argument(
        "--d",
        type=float,
        required=True,
        help="The d parameter for the Deffuant-Weisbuch model.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        required=True,
        help="The mu parameter for the Deffuant-Weisbuch model.",
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="The number of agents in the Deffuant-Weisbuch model.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
        choices=["full", "net", "random", "scale-free"],
        help="The topology of the network in the Deffuant-Weisbuch model.",
    )

    args = parser.parse_args()

    # GA parameters
    pcs = [0.6, 0.7, 0.8]
    pms = [0.05, 0.1, 0.15]
    mutation_ranges = [0.005]
    pop_sizes = [10, 30, 50]

    # SA parameters
    cooling_rates = [0.85, 0.9, 0.95]

    # ML Surrogate parameters
    surrogates = ["GBR", "RFR", "MLP", "XGB"]
    pool_sizes = [256, 512, 1024]
    sample_sizes = [10, 20, 30]

    # Global parameters
    max_iter = 50
    num_of_simulations = 20
    number_of_runs = 1
    stop_fitness = 0.95

    results = []
    o_name = f"o_N{args.N}_d{args.d}_mu{args.mu}_{args.topology}"

    results.extend(
        task_calibration_GS(
            o_name,
            d_bounds=[0.01, 0.5],
            mu_bounds=[0.01, 0.5],
            grid_size=10,
            number_of_runs=number_of_runs,
            num_of_simulations=num_of_simulations,
        )
    )

    results.extend(
        task_calibration_SA(
            o_name,
            d_bounds=[0.01, 0.5],
            mu_bounds=[0.01, 0.5],
            cooling_rates=cooling_rates,
            number_of_runs=number_of_runs,
            num_of_simulations=num_of_simulations,
            max_iter=max_iter,
        )
    )
    results.extend(
        task_calibration_GA1(
            o_name,
            pcs,
            pms,
            mutation_ranges,
            pop_sizes,
            number_of_runs,
            num_of_simulations,
            max_iter,
            stop_fitness,
        )
    )
    results.extend(
        task_calibration_GA2(
            o_name,
            pcs,
            pms,
            mutation_ranges,
            pop_sizes,
            number_of_runs,
            num_of_simulations,
            max_iter,
            stop_fitness,
        )
    )
    for surrogate in surrogates:
        results.extend(
            task_calibration_ML_surrogate(
                o_name,
                surrogate,
                pool_sizes,
                sample_sizes,
                number_of_runs,
                num_of_simulations,
                max_iter,
                stop_fitness,
            )
        )

    with open(f"results/calibration_results_{o_name}.jsonl", "w") as f:
        f.writelines(result + "\n" for result in results)
