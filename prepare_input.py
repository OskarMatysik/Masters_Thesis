import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.models import DeffuantWeisbuchModel


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
        (base_dir / "single_simulations" / "deffuant_weisbuch" / topology).mkdir(
            parents=True, exist_ok=True
        )

    # Multiple simulations structure
    multi_sim_dirs = [
        "multi_deffuant_weisbuch_full",
        "multi_deffuant_weisbuch_net",
        "multi_deffuant_weisbuch_random",
        "multi_deffuant_weisbuch_scale-free",
    ]
    for dir_name in multi_sim_dirs:
        (base_dir / "multiple_simulations" / dir_name).mkdir(
            parents=True, exist_ok=True
        )

    # Results directory
    (base_dir / "results").mkdir(exist_ok=True)


def generate_datasets(topologies: list, Ns: list, ds: list, mus: list):
    """Generate datasets for calibration."""
    for topology in topologies:
        for N in Ns:
            for d in ds:
                for mu in mus:
                    if topology == "full":
                        t = int(10 * (3 + 1 / mu))
                    elif topology == "net":
                        t = int(10 * (3 + 1 / mu) * N / 8)
                    else:
                        t = int(10 * (3 + 1 / mu) * np.sqrt(N))
                    model = DeffuantWeisbuchModel(
                        N=N, d=d, mu=mu, t=t, topology=topology, num_of_data_points=2
                    )
                    model.run()
                    model.export_data()


if __name__ == "__main__":
    # Dataset parameters
    topologies = ["full"]
    Ns = [1000]
    ds = np.arange(0.05, 0.55, 0.05).round(2).tolist()
    mus = np.arange(0.05, 0.55, 0.05).round(2).tolist()

    create_catalog_structure()
    generate_datasets(topologies, Ns, ds, mus)
