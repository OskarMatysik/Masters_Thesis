import sys
from pathlib import Path
from tasks import *

sys.path.insert(0, str(Path(__file__).parent))

from src.models import *
from src.calibration_GA import GA1Calibration


def create_catalog_structure():
    """
    Generate the project directory structure if it doesn't exist.
    
    Creates the following directories:
    - single_simulations/deffuant_weisbuch/{full, net, random, scale-free}/
    - multiple_simulations/{multi_deffuant_weisbuch_full, multi_deffuant_weisbuch_net, 
                           multi_deffuant_weisbuch_random, multi_deffuant_weisbuch_scale-free}/
    - results/
    """
    base_dir = Path(__file__).parent
    
    # Single simulations structure
    single_sim_topologies = ["full", "net", "random", "scale-free"]
    for topology in single_sim_topologies:
        (base_dir / "single_simulations" / "deffuant_weisbuch" / topology).mkdir(parents=True, exist_ok=True)
    
    # Multiple simulations structure
    multi_sim_dirs = [
        "multi_deffuant_weisbuch_full",
        "multi_deffuant_weisbuch_net",
        "multi_deffuant_weisbuch_random",
        "multi_deffuant_weisbuch_scale-free"
    ]
    for dir_name in multi_sim_dirs:
        (base_dir / "multiple_simulations" / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Results directory
    (base_dir / "results").mkdir(exist_ok=True)


if __name__ == "__main__":
    # Dataset parameters
    topologies = ["full"]
    Ns = [1000, 10000, 100000]
    ds = [0.1, 0.25, 0.4]
    mus = [0.1, 0.25, 0.4]

    # GA parameters
    pcs = [0.5, 0.6, 0.7, 0.8, 0.9]
    pms = [0.05, 0.1, 0.15, 0.2, 0.25]
    mutation_ranges = [0.005, 0.01, 0.015, 0.02, 0.025]
    pop_sizes = [10, 20, 30, 40, 50]

    # SA parameters
    cooling_rates = [0.85, 0.875, 0.9, 0.925, 0.95]

    # ML Surrogate parameters
    surrogates = ["GBR", "RFR", "MLP", "XGB"]
    pool_sizes = [256, 512, 1024, 2048, 4096]
    sample_sizes = [10, 20, 30, 40, 50]

    # Global parameters
    max_iter = 100
    num_of_simulations = 20
    number_of_runs = 5
    stop_fitness = 0.95

    create_catalog_structure()

    # Tasks
    generate_datasets(topologies, Ns, ds, mus)
    task_calibration_GA1(o_names = o_names,
                         pcs = pcs,
                         pms = pms,
                         mutation_ranges = mutation_ranges,
                         pop_sizes = pop_sizes,
                         number_of_runs = number_of_runs, 
                         num_of_simulations = num_of_simulations, 
                         max_iter = max_iter, 
                         stop_fitness = stop_fitness
                         )
    task_calibration_GA2(o_names = o_names,
                         pcs = pcs,
                         pms = pms,
                         mutation_ranges = mutation_ranges,
                         pop_sizes = pop_sizes,
                         number_of_runs = number_of_runs,
                         num_of_simulations = num_of_simulations,
                         max_iter = max_iter,
                         stop_fitness = stop_fitness
                         )
    task_calibration_GS(o_names = o_names,
                         d_bounds = [0.01, 0.5],
                         mu_bounds = [0.01, 0.5],
                         grid_size = 50,
                         number_of_runs = number_of_runs,
                         num_of_simulations = num_of_simulations)
    task_calibration_SA(o_names = o_names,
                         d_bounds = [0.01, 0.5],
                         mu_bounds = [0.01, 0.5],
                         cooling_rates = cooling_rates,
                         number_of_runs = number_of_runs,
                         num_of_simulations = num_of_simulations,
                         max_iter = max_iter,
                         stop_fitness = stop_fitness)
    task_calibration_ML_surrogate(
        o_names = o_names,
        surrogates = surrogates,
        pool_sizes = pool_sizes,
        sample_sizes = sample_sizes,
        number_of_runs = number_of_runs,
        num_of_simulations = num_of_simulations,
        max_iter = max_iter,
        stop_fitness = stop_fitness
    )


