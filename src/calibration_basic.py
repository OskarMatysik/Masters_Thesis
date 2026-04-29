from time import time

import numpy as np
import pandas as pd
from scipy.stats import differential_entropy

from .multiple_runs import MultiDW


class GridSearchCalibration:
    def __init__(
        self,
        o_name: str,
        d_bounds: list,
        mu_bounds: list,
        grid_size: int,
        num_of_simulations: int,
        real_d: float,
        real_mu: float,
        topology: str = "full",
        log: bool = False,
    ):
        self.name = o_name
        self.d_bounds = d_bounds
        self.mu_bounds = mu_bounds
        self.grid_size = grid_size
        self.num_of_simulations = num_of_simulations
        self.real_d = real_d
        self.real_mu = real_mu
        self.topology = topology
        self.log = log

        self.t, self.y_real = self._read_opinions(o_name)
        self.d_grid, self.mu_grid = self._generate_grid()
        self.fitness_grid = np.zeros_like(self.d_grid)

        self.total_time = None
        self.abm_calls = 0
        self.best_params = None
        self.best_fitness = None
        self.prediction_error = None

    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T

    def _generate_grid(self):
        """Generate a grid of parameters for calibration."""
        d_values = np.linspace(self.d_bounds[0], self.d_bounds[1], self.grid_size)
        mu_values = np.linspace(self.mu_bounds[0], self.mu_bounds[1], self.grid_size)
        d_grid, mu_grid = np.meshgrid(d_values, mu_values)
        return d_grid.flatten(), mu_grid.flatten()

    def run(self):
        """Run the grid search calibration process."""
        start_time = time()
        for i, (d, mu) in enumerate(zip(self.d_grid, self.mu_grid)):
            if self.log:
                print(
                    f"Evaluating parameters: d={d:.4f}, mu={mu:.4f} ({i + 1}/{len(self.d_grid)})"
                )
            sim = MultiDW(
                N=self.y_real.shape[1],
                d=d,
                mu=mu,
                t=max(self.t),
                topology=self.topology,
                num_of_runs=self.num_of_simulations,
                snapshots=self.t,
            )
            self.abm_calls += self.num_of_simulations
            entropy = sim.run()[-1]
            fitness = self._fitness(entropy)
            self.fitness_grid[i] = fitness

        best_idx = np.argmax(self.fitness_grid)
        self.best_params = np.array([self.d_grid[best_idx], self.mu_grid[best_idx]])
        self.best_fitness = self.fitness_grid[best_idx]
        self.total_time = time() - start_time
        self.prediction_error = np.abs(
            np.array([self.real_d, self.real_mu]) - self.best_params
        )

    def _fitness(self, entropy_pred: list) -> float:
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
        entropy_pred = np.array(entropy_pred)

        return 1 / (1 + np.sum(np.abs(entropy_real - entropy_pred)))

    def export_calibration_results(self):
        """Export calibration results to csv file."""
        df = pd.DataFrame(
            {
                "d": [self.best_params[0]],
                "mu": [self.best_params[1]],
                "fitness": [self.best_fitness],
                "prediction_error": [self.prediction_error],
                "total_time": [self.total_time],
                "abm_calls": [self.abm_calls],
            }
        )
        df.to_csv(f"results/GS_{self.name}.csv", index=False)


class SimulatedAnnealingCalibration:
    def __init__(
        self,
        o_name: str,
        d_bounds: list,
        mu_bounds: list,
        initial_temp: float,
        cooling_rate: float,
        num_of_simulations: int,
        max_iter: int,
        real_d: float,
        real_mu: float,
        param_range: float = 0.05,
        topology: str = "full",
        log: bool = False,
    ):
        self.name = o_name
        self.t, self.y_real = self._read_opinions(o_name)
        self.d_bounds = d_bounds
        self.mu_bounds = mu_bounds
        self.num_of_simulations = num_of_simulations
        self.topology = topology
        self.log = log
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.param_range = param_range
        self.real_d = real_d
        self.real_mu = real_mu

        self.total_time = None
        self.abm_calls = 0
        self.best_params = None
        self.best_fitness = None
        self.prediction_error = None

    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T

    def run(self):
        """Run the simulated annealing calibration process."""
        start_time = time()
        current_d = np.random.uniform(self.d_bounds[0], self.d_bounds[1])
        current_mu = np.random.uniform(self.mu_bounds[0], self.mu_bounds[1])
        sim = MultiDW(
            N=self.y_real.shape[1],
            d=current_d,
            mu=current_mu,
            t=max(self.t),
            topology=self.topology,
            num_of_runs=self.num_of_simulations,
            snapshots=self.t,
        )
        self.abm_calls += self.num_of_simulations
        entropy = sim.run()[-1]
        max_fitness = self._fitness(entropy)
        temp = self.initial_temp
        if self.log:
            print(
                f"Iteration {1}, Temp: {temp:.4f}, Current Fitness: {max_fitness:.4f}"
            )

        for i in range(1, self.max_iter):
            d = np.clip(
                current_d + np.random.normal(0, self.param_range),
                self.d_bounds[0],
                self.d_bounds[1],
            )
            mu = np.clip(
                current_mu + np.random.normal(0, self.param_range),
                self.mu_bounds[0],
                self.mu_bounds[1],
            )
            if self.log:
                print(
                    f"Iteration {i + 1}, Temp: {temp:.4f}, Current Fitness: {max_fitness:.4f}"
                )
            sim = MultiDW(
                N=self.y_real.shape[1],
                d=d,
                mu=mu,
                t=max(self.t),
                topology=self.topology,
                num_of_runs=self.num_of_simulations,
                snapshots=self.t,
            )
            self.abm_calls += self.num_of_simulations
            entropy = sim.run()[-1]
            fitness = self._fitness(entropy)
            if fitness > max_fitness or np.random.rand() < np.exp(
                (fitness - max_fitness) / temp
            ):
                current_d, current_mu, max_fitness = d, mu, fitness

            temp *= self.cooling_rate

        self.best_params = np.array([current_d, current_mu])
        self.best_fitness = max_fitness
        self.total_time = time() - start_time
        self.prediction_error = np.abs(
            np.array([self.real_d, self.real_mu]) - self.best_params
        )

    def _fitness(self, entropy_pred: list) -> float:
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
        entropy_pred = np.array(entropy_pred)

        return 1 / (1 + np.sum(np.abs(entropy_real - entropy_pred)))

    def export_calibration_results(self):
        """Export calibration results to csv file."""
        df = pd.DataFrame(
            {
                "d": [self.best_params[0]],
                "mu": [self.best_params[1]],
                "fitness": [self.best_fitness],
                "prediction_error": [self.prediction_error],
                "total_time": [self.total_time],
                "abm_calls": [self.abm_calls],
            }
        )
        df.to_csv(f"results/SA_{self.name}.csv", index=False)


if __name__ == "__main__":
    # cal = GridSearchCalibration(
    #     o_name = "o_N1000_d0.23_mu0.46_full",
    #     d_bounds = [0.1, 0.5],
    #     mu_bounds = [0.1, 0.5],
    #     grid_size = 15,
    #     num_of_simulations = 5,
    #     topology = "full",
    #     log = True
    #     )
    cal = SimulatedAnnealingCalibration(
        o_name="o_N1000_d0.23_mu0.46_full",
        d_bounds=[0.1, 0.5],
        mu_bounds=[0.1, 0.5],
        initial_temp=1,
        cooling_rate=0.95,
        num_of_simulations=5,
        max_iter=100,
        param_range=0.01,
        topology="full",
        log=True,
    )
    cal.run()
    cal.export_calibration_results()
