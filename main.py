import sys
from pathlib import Path

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
    create_catalog_structure()
    # model_2 = DeffuantWeisbuchModel(N=2000, d=0.2, mu=0.5, t=100, topology="scale-free")
    # model_2.run()
    # model_2.plot_time_chart()
    # model_2.plot_final_vs_initial()
    # model_2.statistics()
    # calibration = GA1Calibration(o_name="o_N1000_d0.2_mu0.5", s_name="s_N1000_d0.2_mu0.5", p_c=0.8, p_m=0.1, max_iter=200, stop_fitness=0.01, L_p=0.1, U_p=0.9)
    # breakpoint()

