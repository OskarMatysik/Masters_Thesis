from argparse import ArgumentParser

from tasks import (
    task_calibration_GA1,
    task_calibration_GA2,
    task_calibration_GS,
    task_calibration_ML_surrogate,
    task_calibration_SA,
)

if __name__ == "__main__":
    # parser = ArgumentParser(
    #     description="Run the main script for dataset generation and calibration tasks."
    # )
    # parser.add_argument(
    #     "--d",
    #     type=float,
    #     required=True,
    #     help="The d parameter for the Deffuant-Weisbuch model.",
    # )
    # parser.add_argument(
    #     "--mu",
    #     type=float,
    #     required=True,
    #     help="The mu parameter for the Deffuant-Weisbuch model.",
    # )
    # parser.add_argument(
    #     "--N",
    #     type=int,
    #     required=True,
    #     help="The number of agents in the Deffuant-Weisbuch model.",
    # )
    # parser.add_argument(
    #     "--topology",
    #     type=str,
    #     required=True,
    #     choices=["full", "net", "random", "scale-free"],
    #     help="The topology of the network in the Deffuant-Weisbuch model.",
    # )

    # args = parser.parse_args()

    # GA parameters
    pcs = [0.7]
    pms = [0.1]
    mutation_ranges = [0.005]
    pop_sizes = [32]

    # SA parameters
    cooling_rates = [0.9]

    # ML Surrogate parameters
    surrogates = ["GBR", "RFR", "MLP", "XGB"]
    pool_sizes = [1024]
    sample_sizes = [20]

    # Global parameters
    num_of_simulations = 3
    number_of_runs = 1
    stop_fitness = 0.9

    max_iter = 100
    results = []

    N, d, mu, topology = 1000, 0.4, 0.4, "full"
    o_name = f"o_N{N}_d{d}_mu{mu}_{topology}"

    # results.extend(
    #     task_calibration_GS(
    #         o_name,
    #         d_bounds=[0.01, 0.5],
    #         mu_bounds=[0.01, 0.5],
    #         grid_sizes=[32],
    #         number_of_runs=number_of_runs,
    #         num_of_simulations=num_of_simulations,
    #     )
    # )
    # results.extend(
    #     task_calibration_SA(
    #         o_name,
    #         d_bounds=[0.05, 0.5],
    #         mu_bounds=[0.05, 0.5],
    #         cooling_rates=cooling_rates,
    #         number_of_runs=number_of_runs,
    #         num_of_simulations=num_of_simulations,
    #         max_iters=[1024],
    #         stop_fitness=stop_fitness
    #     )
    # )
    results.extend(
        task_calibration_GA1(
            o_name,
            pcs=pcs,
            pms=pms,
            mutation_ranges=mutation_ranges,
            pop_sizes=pop_sizes,
            number_of_runs=number_of_runs,
            num_of_simulations=num_of_simulations,
            max_iter=3,
            stop_fitness=stop_fitness,
        )
    )
    # results.extend(
    #     task_calibration_GA2(
    #         o_name,
    #         pcs=pcs,
    #         pms=pms,
    #         mutation_ranges=mutation_ranges,
    #         pop_sizes=pop_sizes,
    #         number_of_runs=number_of_runs,
    #         num_of_simulations=num_of_simulations,
    #         max_iter=32,
    #         stop_fitness=stop_fitness,
    #     )
    # )
    # for surrogate in surrogates:
    #     results.extend(
    #         task_calibration_ML_surrogate(
    #             o_name,
    #             surrogate=surrogate,
    #             pool_sizes=pool_sizes,
    #             sample_sizes=sample_sizes,
    #             number_of_runs=number_of_runs,
    #             num_of_simulations=num_of_simulations,
    #             max_iter=1000, # doesnt matter
    #             stop_fitness=stop_fitness,
    #         )
    #     )

    with open(f"results/calibration_results_{o_name}.jsonl", "w") as f:
        f.writelines(result + "\n" for result in results)
