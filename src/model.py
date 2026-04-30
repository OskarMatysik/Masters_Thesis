from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import differential_entropy


class Model:
    def __init__(
        self,
        o_name: str,
        d_bounds: list[float],
        mu_bounds: list[float],
        num_of_simulations: int,
        real_d: float,
        real_mu: float,
        topology: str,
        log: bool,
    ) -> None:
        self.name = o_name
        self.d_bounds = d_bounds
        self.mu_bounds = mu_bounds
        self.num_of_simulations = num_of_simulations
        self.real_d = real_d
        self.real_mu = real_mu
        self.topology = topology
        self.log = log

        self.t, self.y_real = self._read_opinions(o_name)
        self.prediction_error: float | Any = None
        self.total_time: float | Any = None
        self.abm_calls: int = 0
        self.best_params: NDArray | Any = None
        self.best_fitness: float | Any = None

    @abstractmethod
    def run(self) -> None: ...

    def _read_opinions(self, o_name: str) -> tuple[NDArray, NDArray]:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T

    def _fitness(self, entropy_pred: list[float]) -> float:
        """
        Calculate fitness based on MSE between real and predicted CDFs for single simulation.
        y_pred (np.ndarray): Predicted opinions for the given time steps.

        1. ecdf approach - failed (add y_pred: np.ndarray = None to work)
        2. entropy calculation approach - good but slow (add y_pred: np.ndarray = None to work)
        3. entropy from multidw - current solution - entropy of y_pred instead of raw y_pred
        """
        entropy_real = np.array(
            [differential_entropy(sample) for sample in self.y_real]
        )

        return 1 / (1 + np.sum(np.abs(entropy_real - np.array(entropy_pred))))
