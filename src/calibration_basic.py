from __future__ import annotations

from time import time
from typing import Any, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .model import Model
from .multiple_runs import MultiDW


class GridSearchCalibration(Model):
    def __init__(
        self,
        o_name: str,
        d_bounds: list[float],
        mu_bounds: list[float],
        num_of_simulations: int,
        real_d: float,
        real_mu: float,
        topology: str,
        grid_size: int,
        log: bool = False,
    ):
        super().__init__(
            o_name=o_name,
            d_bounds=d_bounds,
            mu_bounds=mu_bounds,
            num_of_simulations=num_of_simulations,
            real_d=real_d,
            real_mu=real_mu,
            topology=topology,
            log=log,
        )

        self.grid_size = grid_size
        self.d_grid, self.mu_grid = self._generate_grid()
        self.fitness_grid = np.zeros_like(self.d_grid)

    def _generate_grid(self):
        """Generate a grid of parameters for calibration."""
        d_values = np.linspace(self.d_bounds[0], self.d_bounds[1], self.grid_size)
        mu_values = np.linspace(self.mu_bounds[0], self.mu_bounds[1], self.grid_size)
        d_grid, mu_grid = np.meshgrid(d_values, mu_values)
        return d_grid.flatten(), mu_grid.flatten()

    @override
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
                t=np.max(
                    self.t,
                ).astype(int),
                topology=self.topology,
                num_of_runs=self.num_of_simulations,
                snapshots=self.t.tolist(),
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


class SimulatedAnnealingCalibration(Model):
    def __init__(
        self,
        o_name: str,
        d_bounds: list[float],
        mu_bounds: list[float],
        num_of_simulations: int,
        real_d: float,
        real_mu: float,
        topology: str,
        initial_temp: float,
        cooling_rate: float,
        max_iter: int,
        param_range: float = 0.05,
        log: bool = False,
    ):
        super().__init__(
            o_name=o_name,
            d_bounds=d_bounds,
            mu_bounds=mu_bounds,
            num_of_simulations=num_of_simulations,
            real_d=real_d,
            real_mu=real_mu,
            topology=topology,
            log=log,
        )
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.param_range = param_range

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
            snapshots=self.t.tolist(),
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
                snapshots=self.t.tolist(),
            )
            self.abm_calls += self.num_of_simulations
            entropy = sim.run()[-1]
            fitness = self._fitness(entropy)
            if fitness > max_fitness or np.random.rand() < np.exp(
                (fitness - max_fitness) / temp
            ):
                current_d, current_mu, max_fitness = d, mu, fitness

            temp *= self.cooling_rate

        self.best_params: NDArray | Any = np.array([current_d, current_mu])
        self.best_fitness = max_fitness
        self.total_time = time() - start_time
        self.prediction_error = np.abs(
            np.array([self.real_d, self.real_mu]) - self.best_params
        )

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
