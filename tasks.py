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

def task_calibration_GA1(o_names: list, pcs: list, pms: list, mutation_ranges: list, pop_sizes: list, 
                         number_of_runs=1, num_of_simulations: int=100):
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
                                max_iter = 50,
                                stop_fitness = 0.95,
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
                         number_of_runs=1, num_of_simulations: int=100):
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
                                max_iter = 50,
                                stop_fitness = 0.95,
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

if __name__ == "__main__":

    topologies = ["full"]
    Ns = [1000]
    ds = [0.2, 0.4]
    mus = [0.4]
    generate_datasets(topologies, Ns, ds, mus)

    o_names = [f"o_N{N}_d{d}_mu{mu}_{topology}" for topology in topologies for N in Ns for d in ds for mu in mus]
    pcs = [0.7, 0.9]
    pms = [0.05, 0.1]
    mutation_ranges = [0.005]
    pop_sizes = [20, 50]

    # task_calibration_GA1(o_names, pcs, pms, mutation_ranges, pop_sizes, number_of_runs=1, num_of_simulations=3)
    task_calibration_GA2(o_names, pcs, pms, mutation_ranges, pop_sizes, number_of_runs=1, num_of_simulations=3)

