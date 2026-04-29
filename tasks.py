import json
from itertools import product

import numpy as np

from src.calibration_basic import GridSearchCalibration, SimulatedAnnealingCalibration
from src.calibration_GA import GA1Calibration, GA2Calibration
from src.calibration_surrogate import MLSurrogateCalibration


def task_calibration_GS(
    o_name: str,
    d_bounds: list,
    mu_bounds: list,
    grid_size: int,
    number_of_runs: int = 1,
    num_of_simulations: int = 100,
) -> list[str]:

    d_real = float(o_name.split("_")[2][1:])
    mu_real = float(o_name.split("_")[3][2:])
    topology = o_name.split("_")[4]

    ds = []
    mus = []
    fitnesses = []
    prediction_errors = []
    total_times = []
    abm_calls = []

    for i in range(number_of_runs):
        cal = GridSearchCalibration(
            o_name=o_name,
            d_bounds=d_bounds,
            mu_bounds=mu_bounds,
            grid_size=grid_size,
            num_of_simulations=num_of_simulations,
            topology=topology,
            real_d=d_real,
            real_mu=mu_real,
        )
        cal.run()
        ds.append(cal.best_params[0])
        mus.append(cal.best_params[1])
        fitnesses.append(cal.best_fitness)
        prediction_errors.append(cal.prediction_error)
        total_times.append(cal.total_time)
        abm_calls.append(cal.abm_calls)
    return [
        json.dumps(
            {
                "model": "GS",
                "d": np.mean(ds),
                "mu": np.mean(mus),
                "fitness": np.mean(fitnesses),
                "prediction_error": np.mean(prediction_errors),
                "total_time": np.mean(total_times),
                "abm_calls": np.mean(abm_calls),
            },
            indent=0,
        ).replace("\n", "")
    ]


def task_calibration_SA(
    o_name: str,
    d_bounds: list,
    mu_bounds: list,
    cooling_rates: list,
    number_of_runs: int = 1,
    num_of_simulations: int = 100,
    max_iter: int = 100,
) -> list[str]:
    d_real = float(o_name.split("_")[2][1:])
    mu_real = float(o_name.split("_")[3][2:])
    topology = o_name.split("_")[4]
    results = []

    for cooling_rate in cooling_rates:
        ds = []
        mus = []
        fitnesses = []
        prediction_errors = []
        total_times = []
        abm_calls = []

        print(f"Running Simulated Annealing with cooling_rate={cooling_rate}")
        for i in range(number_of_runs):
            cal = SimulatedAnnealingCalibration(
                o_name=o_name,
                d_bounds=d_bounds,
                mu_bounds=mu_bounds,
                initial_temp=1,
                cooling_rate=cooling_rate,
                num_of_simulations=num_of_simulations,
                max_iter=max_iter,
                topology=topology,
                real_d=d_real,
                real_mu=mu_real,
            )
            cal.run()
            ds.append(cal.best_params[0])
            mus.append(cal.best_params[1])
            fitnesses.append(cal.best_fitness)
            prediction_errors.append(cal.prediction_error)
            total_times.append(cal.total_time)
            abm_calls.append(cal.abm_calls)
        results.append(
            json.dumps(
                {
                    "model": "SA",
                    "cooling_rate": cooling_rate,
                    "d": np.mean(ds),
                    "mu": np.mean(mus),
                    "fitness": np.mean(fitnesses),
                    "prediction_error": np.mean(prediction_errors),
                    "total_time": np.mean(total_times),
                    "abm_calls": np.mean(abm_calls),
                },
                indent=0,
            ).replace("\n", "")
        )
    return results


def task_calibration_GA1(
    o_name: str,
    pcs: list,
    pms: list,
    mutation_ranges: list,
    pop_sizes: list,
    number_of_runs=1,
    num_of_simulations: int = 100,
    max_iter: int = 100,
    stop_fitness: float = 0.95,
) -> list[str]:
    d_real = float(o_name.split("_")[2][1:])
    mu_real = float(o_name.split("_")[3][2:])
    topology = o_name.split("_")[4]
    results = []
    for pc, pm, mutation_range, pop_size in product(
        pcs, pms, mutation_ranges, pop_sizes
    ):
        ds = []
        mus = []
        fitnesses = []
        prediction_errors = []
        total_times = []
        abm_calls = []
        for i in range(number_of_runs):
            print(
                f"Running GA1 with pc={pc}, pm={pm}, mutation_range={mutation_range}, pop_size={pop_size}"
            )
            cal = GA1Calibration(
                o_name=o_name,
                num_of_params=2,
                pop_size=pop_size,
                p_c=pc,
                p_m=pm,
                max_iter=max_iter,
                stop_fitness=stop_fitness,
                L_p=[0, 0],
                U_p=[0.5, 0.5],
                mutation_range=mutation_range,
                topology=topology,
                num_of_simulations=num_of_simulations,
                d_real=d_real,
                mu_real=mu_real,
            )
            cal.run()
            ds.append(cal.result[np.argmax(cal.fitness)][0])
            mus.append(cal.result[np.argmax(cal.fitness)][1])
            fitnesses.append(max(cal.fitness))
            prediction_errors.append(cal.prediction_error)
            total_times.append(cal.total_time)
            abm_calls.append(cal.abm_calls)
        results.append(
            json.dumps(
                {
                    "model": "GA1",
                    "pc": pc,
                    "pm": pm,
                    "mutation_range": mutation_range,
                    "pop_size": pop_size,
                    "d": np.mean(ds),
                    "mu": np.mean(mus),
                    "fitness": np.mean(fitnesses),
                    "prediction_error": np.mean(prediction_errors),
                    "total_time": np.mean(total_times),
                    "abm_calls": np.mean(abm_calls),
                },
                indent=0,
            ).replace("\n", "")
        )
    return results


def task_calibration_GA2(
    o_name: str,
    pcs: list,
    pms: list,
    mutation_ranges: list,
    pop_sizes: list,
    number_of_runs=1,
    num_of_simulations: int = 100,
    max_iter: int = 50,
    stop_fitness: float = 0.95,
) -> list[str]:
    d_real = float(o_name.split("_")[2][1:])
    mu_real = float(o_name.split("_")[3][2:])
    topology = o_name.split("_")[4]
    results = []

    for pc, pm, mutation_range, pop_size in product(
        pcs, pms, mutation_ranges, pop_sizes
    ):
        ds = []
        mus = []
        fitnesses = []
        prediction_errors = []
        total_times = []
        abm_calls = []
        print(
            f"Running GA2 with pc={pc}, pm={pm}, mutation_range={mutation_range}, pop_size={pop_size}"
        )
        for i in range(number_of_runs):
            cal = GA2Calibration(
                o_name=o_name,
                num_of_params=2,
                pop_size=pop_size,
                p_c=pc,
                p_m=pm,
                max_iter=max_iter,
                stop_fitness=stop_fitness,
                L_p=[0.01, 0.01],
                U_p=[0.5, 0.5],
                mutation_range=mutation_range,
                topology=topology,
                num_of_simulations=num_of_simulations,
                d_real=d_real,
                mu_real=mu_real,
                beta=6,
                gamma_L=2,
                gamma_U=10,
                alpha=0.2,
            )
            cal.run()
            ds.append(cal.result[np.argmax(cal.fitness)][0])
            mus.append(cal.result[np.argmax(cal.fitness)][1])
            fitnesses.append(max(cal.fitness))
            prediction_errors.append(cal.prediction_error)
            total_times.append(cal.total_time)
            abm_calls.append(cal.abm_calls)
        results.append(
            json.dumps(
                {
                    "model": "GA2",
                    "pc": pc,
                    "pm": pm,
                    "mutation_range": mutation_range,
                    "pop_size": pop_size,
                    "d": np.mean(ds),
                    "mu": np.mean(mus),
                    "fitness": np.mean(fitnesses),
                    "prediction_error": np.mean(prediction_errors),
                    "total_time": np.mean(total_times),
                    "abm_calls": np.mean(abm_calls),
                },
                indent=0,
            ).replace("\n", "")
        )
    return results


def task_calibration_ML_surrogate(
    o_name: str,
    surrogate: str,
    pool_sizes: list,
    sample_sizes: list,
    number_of_runs=1,
    num_of_simulations: int = 100,
    max_iter: int = 50,
    stop_fitness: float = 0.95,
) -> list[str]:
    d_real = float(o_name.split("_")[2][1:])
    mu_real = float(o_name.split("_")[3][2:])
    topology = o_name.split("_")[4]
    results = []

    for pool_size in pool_sizes:
        for sample_size in sample_sizes:
            ds = []
            mus = []
            fitnesses = []
            prediction_errors = []
            total_times = []
            abm_calls = []
            print(
                f"Running ML Surrogate Calibration, surrogate={surrogate}, pool_size={pool_size}, sample_size={sample_size}"
            )
            for i in range(number_of_runs):
                cal = MLSurrogateCalibration(
                    o_name=o_name,
                    surrogate=surrogate,
                    sampling_method="Sobol",
                    pool_size=pool_size,
                    sample_size=sample_size,
                    max_iter=max_iter,
                    stop_fitness=stop_fitness,
                    num_of_simulations=num_of_simulations,
                    topology=topology,
                    d_real=d_real,
                    mu_real=mu_real,
                )
                cal.run()
                ds.append(cal.best_params[0])
                mus.append(cal.best_params[1])
                fitnesses.append(cal.best_fitness)
                prediction_errors.append(cal.prediction_error)
                total_times.append(cal.total_time)
                abm_calls.append(cal.abm_calls)
            results.append(
                json.dumps(
                    {
                        "model": surrogate,
                        "pool_size": pool_size,
                        "sample_size": sample_size,
                        "d": np.mean(ds),
                        "mu": np.mean(mus),
                        "fitness": np.mean(fitnesses),
                        "prediction_error": np.mean(prediction_errors),
                        "total_time": np.mean(total_times),
                        "abm_calls": np.mean(abm_calls),
                    },
                    indent=0,
                ).replace("\n", "")
            )
    return results
