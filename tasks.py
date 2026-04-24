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

def task_calibration_GA1(o_names: list, pcs: list, pms: list, mutation_ranges: list, pop_sizes: list, number_of_runs=1):
    for o_name in o_names:
        d_real = float(o_name.split("_")[2][1:])
        mu_real = float(o_name.split("_")[3][2:])
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
                            cal = GA1Calibration(
                                o_name = o_name,
                                num_of_params = 2,
                                pop_size = pop_size,
                                p_c = pc,
                                p_m = pm,
                                max_iter = 50,
                                stop_fitness = 0.95,
                                L_p = 0.1,
                                U_p = 0.5,
                                mutation_range = mutation_range,
                                topology = "full",
                                num_of_simulations = 5,
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
                        df.append({
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
                        }, ignore_index=True)
        df.to_csv(f"results/results_GA1_{o_name}.csv", index=False)    


def task_calibration_GA2(o_names: list, pcs: list, pms: list, mutation_ranges: list, pop_sizes: list, number_of_runs=1):
    for o_name in o_names:
        d_real = float(o_name.split("_")[2][1:])
        mu_real = float(o_name.split("_")[3][2:])
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
                            cal = GA2Calibration(
                                o_name = o_name,
                                num_of_params = 2,
                                pop_size = pop_size,
                                p_c = pc,
                                p_m = pm,
                                max_iter = 50,
                                stop_fitness = 0.95,
                                L_p = 0.1,
                                U_p = 0.5,
                                mutation_range = mutation_range,
                                topology = "full",
                                num_of_simulations = 5,
                                d_real = d_real,
                                mu_real = mu_real,
                            )
                            cal.run()
                        df.append({
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
                        }, ignore_index=True)
        df.to_csv(f"results/results_GA2_{o_name}.csv", index=False)   
