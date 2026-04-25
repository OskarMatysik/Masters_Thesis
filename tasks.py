from src.calibration_basic import GridSearchCalibration, SimulatedAnnealingCalibration
from src.calibration_GA import GA1Calibration, GA2Calibration
from src.calibration_surrogate import MLSurrogateCalibration

from src.models import DeffuantWeisbuchModel
import numpy as np
import pandas as pd

def generate_datasets(topologies: list, Ns: list, ds: list, mus:list):
    """Generate datasets for calibration."""
    for topology in topologies:
        for N in Ns:
            for d in ds:
                for mu in mus:
                    if topology == "full":
                        t = int(10 * (3 + 1/mu))
                    elif topology == "net":
                        t = int(10 * (3 + 1/mu) * N/8)
                    else:
                        t = int(10 * (3 + 1/mu) * np.sqrt(N))
                    model = DeffuantWeisbuchModel(N=N, d=d, mu=mu, t=t, topology=topology, num_of_data_points=2)
                    model.run()
                    model.export_data()

def task_calibration_GS(o_names: list, d_bounds: list, mu_bounds: list, grid_size: int, 
                        number_of_runs=1, num_of_simulations: int=100):
    for o_name in o_names:
        d_real = float(o_name.split("_")[2][1:])
        mu_real = float(o_name.split("_")[3][2:])
        topology = o_name.split("_")[4]
        df = pd.DataFrame(columns=["d", 
                                   "mu", 
                                   "fitness", 
                                   "prediction_error", 
                                   "total_time", 
                                   "abm_calls"])
        ds = []
        mus = []
        fitnesses = []
        prediction_errors = []
        total_times = []
        abm_calls = []
        print(f"Running Grid Search for {o_name}")
        for i in range(number_of_runs):
            cal = GridSearchCalibration(
                o_name = o_name,
                d_bounds = d_bounds,
                mu_bounds = mu_bounds,
                grid_size = grid_size,
                num_of_simulations = num_of_simulations,
                topology = topology,
                real_d = d_real,
                real_mu = mu_real
            )
            cal.run()
            ds.append(cal.best_params[0])
            mus.append(cal.best_params[1])
            fitnesses.append(cal.best_fitness)
            prediction_errors.append(cal.prediction_error)
            total_times.append(cal.total_time)
            abm_calls.append(cal.abm_calls)
        df.loc[len(df)] = {
            "d": np.mean(ds),
            "mu": np.mean(mus),
            "fitness": np.mean(fitnesses),
            "prediction_error": np.mean(prediction_errors),
            "total_time": np.mean(total_times),
            "abm_calls": np.mean(abm_calls)
        }
        df.to_csv(f"results/results_GS_{o_name}.csv", index=False)    


def task_calibration_SA(o_names: list, d_bounds: list, mu_bounds: list, cooling_rates: list,
                        number_of_runs=1, num_of_simulations: int=100, max_iter: int=100, stop_fitness: float=0.95):
    for o_name in o_names:
        d_real = float(o_name.split("_")[2][1:])
        mu_real = float(o_name.split("_")[3][2:])
        topology = o_name.split("_")[4]
        df = pd.DataFrame(columns=[
                                "cooling_rate",
                                "d", 
                                "mu", 
                                "fitness", 
                                "prediction_error", 
                                "total_time", 
                                "abm_calls"])
        for cooling_rate in cooling_rates:
            ds = []
            mus = []
            fitnesses = []
            prediction_errors = []
            total_times = []
            abm_calls = []
            print(f"Running Simulated Annealing for {o_name}, cooling_rates={cooling_rate}")
            for i in range(number_of_runs):
                cal = SimulatedAnnealingCalibration(
                    o_name = o_name,
                    d_bounds = d_bounds,
                    mu_bounds = mu_bounds,
                    initial_temp = 1,
                    cooling_rate = cooling_rate,
                    num_of_simulations = num_of_simulations,
                    max_iter = max_iter,
                    stop_fitness = stop_fitness,
                    topology = topology,
                    real_d = d_real,
                    real_mu = mu_real,
                )
                cal.run()
                ds.append(cal.best_params[0])
                mus.append(cal.best_params[1])
                fitnesses.append(cal.best_fitness)
                prediction_errors.append(cal.prediction_error)
                total_times.append(cal.total_time)
                abm_calls.append(cal.abm_calls)
            df.loc[len(df)] = {
                "cooling_rate": cooling_rate,
                "d": np.mean(ds),
                "mu": np.mean(mus),
                "fitness": np.mean(fitnesses),
                "prediction_error": np.mean(prediction_errors),
                "total_time": np.mean(total_times),
                "abm_calls": np.mean(abm_calls)
            }
        df.to_csv(f"results/results_SA_{o_name}.csv", index=False)    


def task_calibration_GA1(o_names: list, pcs: list, pms: list, mutation_ranges: list, pop_sizes: list, 
                         number_of_runs=1, num_of_simulations: int=100, max_iter: int=100, stop_fitness: float=0.95):
    for o_name in o_names:
        d_real = float(o_name.split("_")[2][1:])
        mu_real = float(o_name.split("_")[3][2:])
        topology = o_name.split("_")[4]
        df = pd.DataFrame(columns=["pc", 
                                   "pm", 
                                   "mutation_range", 
                                   "pop_size", 
                                   "d", 
                                   "mu", 
                                   "fitness", 
                                   "prediction_error", 
                                   "total_time", 
                                   "abm_calls"])
        for pc in pcs:
            for pm in pms:
                for mutation_range in mutation_ranges:
                    for pop_size in pop_sizes:
                        ds = []
                        mus = []
                        fitnesses = []
                        prediction_errors = []
                        total_times = []
                        abm_calls = []
                        for i in range(number_of_runs):
                            print(f"Running GA1 for {o_name} with pc={pc}, pm={pm}, mutation_range={mutation_range}, pop_size={pop_size}")
                            cal = GA1Calibration(
                                o_name = o_name,
                                num_of_params = 2,
                                pop_size = pop_size,
                                p_c = pc,
                                p_m = pm,
                                max_iter = max_iter,
                                stop_fitness = stop_fitness,
                                L_p = [0, 0],
                                U_p = [0.5, 0.5],
                                mutation_range = mutation_range,
                                topology = topology,
                                num_of_simulations = num_of_simulations,
                                d_real = d_real,
                                mu_real = mu_real,
                            )
                            cal.run()
                            ds.append(cal.result[np.argmax(cal.fitness)][0])
                            mus.append(cal.result[np.argmax(cal.fitness)][1])
                            fitnesses.append(max(cal.fitness))
                            prediction_errors.append(cal.prediction_error)
                            total_times.append(cal.total_time)
                            abm_calls.append(cal.abm_calls)
                        df.loc[len(df)] = {
                            "pc": pc,
                            "pm": pm,
                            "mutation_range": mutation_range,
                            "pop_size": pop_size,
                            "d": np.mean(ds),
                            "mu": np.mean(mus),
                            "fitness": np.mean(fitnesses),
                            "prediction_error": np.mean(prediction_errors),
                            "total_time": np.mean(total_times),
                            "abm_calls": np.mean(abm_calls)
                        }
        df.to_csv(f"results/results_GA1_{o_name}.csv", index=False)    


def task_calibration_GA2(o_names: list, pcs: list, pms: list, mutation_ranges: list, pop_sizes: list, 
                         number_of_runs=1, num_of_simulations: int = 100, max_iter: int = 50, stop_fitness: float = 0.95):
    for o_name in o_names:
        d_real = float(o_name.split("_")[2][1:])
        mu_real = float(o_name.split("_")[3][2:])
        topology = o_name.split("_")[4]
        df = pd.DataFrame(columns=["pc", 
                                   "pm", 
                                   "mutation_range", 
                                   "pop_size", 
                                   "d", 
                                   "mu", 
                                   "fitness", 
                                   "prediction_error", 
                                   "total_time", 
                                   "abm_calls"])
        for pc in pcs:
            for pm in pms:
                for mutation_range in mutation_ranges:
                    for pop_size in pop_sizes:
                        ds = []
                        mus = []
                        fitnesses = []
                        prediction_errors = []
                        total_times = []
                        abm_calls = []
                        print(f"Running GA2 for {o_name} with pc={pc}, pm={pm}, mutation_range={mutation_range}, pop_size={pop_size}")
                        for i in range(number_of_runs):
                            cal = GA2Calibration(
                                o_name = o_name,
                                num_of_params = 2,
                                pop_size = pop_size,
                                p_c = pc,
                                p_m = pm,
                                max_iter = max_iter,
                                stop_fitness = stop_fitness,
                                L_p = [0.01, 0.01],
                                U_p = [0.5, 0.5],
                                mutation_range = mutation_range,
                                topology = topology,
                                num_of_simulations = num_of_simulations,
                                d_real = d_real,
                                mu_real = mu_real,
                                beta = 6,
                                gamma_L = 2,
                                gamma_U = 10,
                                alpha = 0.2
                            )
                            cal.run()
                            ds.append(cal.result[np.argmax(cal.fitness)][0])
                            mus.append(cal.result[np.argmax(cal.fitness)][1])
                            fitnesses.append(max(cal.fitness))
                            prediction_errors.append(cal.prediction_error)
                            total_times.append(cal.total_time)
                            abm_calls.append(cal.abm_calls)
                        df.loc[len(df)] = {
                            "pc": pc,
                            "pm": pm,
                            "mutation_range": mutation_range,
                            "pop_size": pop_size,
                            "d": np.mean(ds),
                            "mu": np.mean(mus),
                            "fitness": np.mean(fitnesses),
                            "prediction_error": np.mean(prediction_errors),
                            "total_time": np.mean(total_times),
                            "abm_calls": np.mean(abm_calls)
                        }
        df.to_csv(f"results/results_GA2_{o_name}.csv", index=False)   

def task_calibration_ML_surrogate(o_names: list, surrogates: list, pool_sizes: list, sample_sizes: list,
                                 number_of_runs=1, num_of_simulations: int=100, max_iter: int=50, stop_fitness: float=0.95):
    for o_name in o_names:
        d_real = float(o_name.split("_")[2][1:])
        mu_real = float(o_name.split("_")[3][2:])
        topology = o_name.split("_")[4]
        for surrogate in surrogates:
            df = pd.DataFrame(columns=["pool_size",
                                    "sample_size", 
                                    "d", 
                                    "mu", 
                                    "fitness", 
                                    "prediction_error", 
                                    "total_time", 
                                    "abm_calls"])
            for pool_size in pool_sizes:
                for sample_size in sample_sizes:
                    ds = []
                    mus = []
                    fitnesses = []
                    prediction_errors = []
                    total_times = []
                    abm_calls = []
                    print(f"Running ML Surrogate Calibration for {o_name}, surrogate={surrogate}, pool_size={pool_size}, sample_size={sample_size}")
                    for i in range(number_of_runs):
                        cal = MLSurrogateCalibration(
                            o_name = o_name,
                            surrogate = surrogate,
                            sampling_method = "Sobol",
                            pool_size = pool_size,
                            sample_size = sample_size,
                            max_iter = max_iter,
                            stop_fitness = stop_fitness,
                            num_of_simulations = num_of_simulations,
                            topology = topology,
                            d_real = d_real,
                            mu_real = mu_real,
                        )
                        cal.run()
                        ds.append(cal.best_params[0])
                        mus.append(cal.best_params[1])
                        fitnesses.append(cal.best_fitness)
                        prediction_errors.append(cal.prediction_error)
                        total_times.append(cal.total_time)
                        abm_calls.append(cal.abm_calls)
                    df.loc[len(df)] = {
                        "pool_size": pool_size,
                        "sample_size": sample_size,
                        "d": np.mean(ds),
                        "mu": np.mean(mus),
                        "fitness": np.mean(fitnesses),
                        "prediction_error": np.mean(prediction_errors),
                        "total_time": np.mean(total_times),
                        "abm_calls": np.mean(abm_calls)
                    }
            df.to_csv(f"results/results_{surrogate}_{o_name}.csv", index=False)

if __name__ == "__main__":

    topologies = ["full"]
    Ns = [1000]
    ds = [0.2, 0.4]
    mus = [0.4]
    generate_datasets(topologies, Ns, ds, mus)

    o_names = [f"o_N{N}_d{d}_mu{mu}_{topology}" for topology in topologies for N in Ns for d in ds for mu in mus]
    
    # GA parameter sets
    pcs = [0.7, 0.9]
    pms = [0.05, 0.1]
    mutation_ranges = [0.005]
    pop_sizes = [20, 50]

    # SA parameter sets
    cooling_rates = [0.85, 0.9, 0.95]

    #ML Surrogate parameter sets
    surrogates = ["GBR", "RFR", "MLP", "XGB"]
    pool_sizes = [512, 1024, 2048]
    sample_sizes = [20, 50]


    # task_calibration_GA1(o_names, pcs, pms, mutation_ranges, pop_sizes, number_of_runs=1, num_of_simulations=3)
    # task_calibration_GA2(o_names, pcs, pms, mutation_ranges, pop_sizes, number_of_runs=1, num_of_simulations=3)
    # task_calibration_GS(o_names, d_bounds=[0.01, 0.5], mu_bounds=[0.01, 0.5], grid_size=10, number_of_runs=1, num_of_simulations=3)
    # task_calibration_SA(o_names, d_bounds=[0.01, 0.5], mu_bounds=[0.01, 0.5], cooling_rates=cooling_rates, number_of_runs=1, num_of_simulations=3)
    task_calibration_ML_surrogate(o_names, surrogates, pool_sizes, sample_sizes, number_of_runs=1, num_of_simulations=3)