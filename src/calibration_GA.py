from time import time
from typing import override

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import differential_entropy

from .model import Model
from .multiple_runs import MultiDW


class GACalibration(Model):
    def __init__(
        self,
        o_name: str,
        d_bounds: list[float],
        mu_bounds: list[float],
        num_of_simulations: int,
        real_d: float,
        real_mu: float,
        topology: str,
        max_iter: int,
        stop_fitness: float,
        mutation_range: float,
        pop_size: int,
        p_c: float,
        p_m: float,
        log: bool = False,
    ) -> None:
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

        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.stop_fitness = stop_fitness
        self.mutation_range = np.sqrt(mutation_range)

    def _init_population(self) -> np.ndarray:
        """Initialize the population for the genetic algorithm.
        Returns array with rows being chromosomes."""
        parameters = []
        for _ in range(self.pop_size):
            parameters.append(
                [
                    np.random.uniform(self.d_bounds[0], self.d_bounds[1]),
                    np.random.uniform(self.mu_bounds[0], self.mu_bounds[1]),
                ]
            )

        population = np.vstack(parameters)
        return population

    def _crossover(self, parents: NDArray) -> tuple:
        """Perform crossover for 2 chosen chromosomes."""
        gene = np.random.choice(2)
        ratio = np.random.random()
        chromosome_1, chromosome_2 = parents[0], parents[1]
        chromosome_1[gene], chromosome_2[gene] = (
            chromosome_1[gene] * ratio + chromosome_2[gene] * (1 - ratio),
            chromosome_2[gene] * ratio + chromosome_1[gene] * (1 - ratio),
        )

        return chromosome_1, chromosome_2


class GA1Calibration(GACalibration):
    """Genetic algorithm calibration using the Deffuant–Weisbuch model.

    This class performs GA-based parameter search by comparing model outcomes
    to real opinion data using entropy-based fitness evaluation.
    """

    @override
    def run(self) -> None:
        """Run the genetic algorithm to calibrate the model."""
        start_time = time()
        new_population = self._init_population()
        i = 0
        fitness_values = np.zeros(self.pop_size)
        population: NDArray | None = None
        for _ in range(self.max_iter):
            population = np.array(new_population)
            new_population = []

            for chr_id, chromosome in enumerate(population):
                multi_model = MultiDW(
                    self.num_of_simulations,
                    N=np.shape(self.y_real)[1],
                    d=chromosome[0],
                    mu=chromosome[1],
                    t=max(self.t),
                    topology=self.topology,
                    snapshots=self.t.tolist(),
                )
                self.abm_calls += self.num_of_simulations
                entropy_pred = multi_model.run()[-1]
                fitness_values[chr_id] = self._fitness(entropy_pred)

            for _ in range(int(self.pop_size * self.p_c // 2)):
                tournament = np.sort(np.random.choice(self.pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                weakest_index = np.argmin(f_vals)  # this is index of the weakest
                parents_ids = np.delete(tournament, weakest_index)
                parents = population[parents_ids]
                new_population.extend(self._crossover(parents))

            for _ in range(self.pop_size - int(self.pop_size * self.p_c // 2) * 2):
                tournament = np.sort(np.random.choice(self.pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                strongest_index = np.argmax(f_vals)
                strongest = population[tournament[strongest_index]]
                new_population.append(strongest)

            for chr_id in range(len(new_population)):
                if np.random.random() < self.p_m:
                    new_value_d = new_population[chr_id][0] + np.random.normal(
                        0, self.mutation_range
                    )
                    new_population[chr_id][0] = np.clip(
                        new_value_d, self.d_bounds[0], self.d_bounds[1]
                    )
                if np.random.random() < self.p_m:
                    new_value_mu = new_population[chr_id][1] + np.random.normal(
                        0, self.mutation_range
                    )
                    new_population[chr_id][1] = np.clip(
                        new_value_mu, self.mu_bounds[0], self.mu_bounds[1]
                    )

            if self.log:
                print(f"Iteration: {i}, Best fit: {max(fitness_values)}")
            if max(fitness_values) <= self.stop_fitness:
                break

        assert population is not None
        self.best_fitness = np.max(fitness_values)
        self.best_params = population[np.argmax(fitness_values)]
        self.total_time = time() - start_time
        self.prediction_error = np.abs(
            np.array([self.real_d, self.real_mu]) - self.best_params
        )

    def export_calibration_results(self):
        """Return calibration results to csv file."""
        df = pd.DataFrame(
            {
                "d": self.best_params[0],
                "mu": self.best_params[1],
                "fitness": self.best_fitness,
                "prediction_error": self.prediction_error,
                "total_time": self.total_time,
                "abm_calls": self.abm_calls,
            }
        )
        df.to_csv(f"results/GA1_{self.name}.csv", index=False)


class GA2Calibration(GACalibration):
    """Genetic algorithm calibration with dynamic thresholding and duplicate removal.

    This class extends the GA1-style calibration to include parameter rounding,
    adaptive gamma thresholding, and duplicate chromosome pruning.
    """

    def __init__(
        self,
        o_name: str,
        d_bounds: list[float],
        mu_bounds: list[float],
        num_of_simulations: int,
        real_d: float,
        real_mu: float,
        topology: str,
        pop_size: int,
        p_c: float,
        p_m: float,
        max_iter: int,
        stop_fitness: float,
        mutation_range: float,
        beta: int,
        gamma_L: float,
        gamma_U: float,
        alpha: float,
        log: bool = False,
    ) -> None:
        """Initialize the GA2 calibration.

        Args:
            o_name (str): Opinion data file name without extension.
            pop_size (int): Number of chromosomes in the population.
            p_c (float): Probability of performing crossover.
            p_m (float): Probability of mutating each gene.
            max_iter (int): Maximum number of algorithm iterations.
            stop_fitness (float): Fitness threshold for early stopping.
            L_p (Sequence[float]): Lower bounds for each parameter.
            U_p (Sequence[float]): Upper bounds for each parameter.
            mutation_range (float): Variance used to compute mutation standard deviation.
            num_of_simulations (int): Number of simulation runs used per chromosome.
            topology (str): Network topology for the model simulations.
            beta (int): Number of decimal places used for rounding chromosome values.
            gamma_L (float): Lower bound for the dynamic gamma threshold.
            gamma_U (float): Upper bound for the dynamic gamma threshold.
            alpha (float): Decay rate for updating gamma over iterations.
            log (bool): If True, print progress updates during optimization.
        """
        super().__init__(
            o_name=o_name,
            d_bounds=d_bounds,
            mu_bounds=mu_bounds,
            num_of_simulations=num_of_simulations,
            real_d=real_d,
            real_mu=real_mu,
            topology=topology,
            log=log,
            pop_size=pop_size,
            p_c=p_c,
            p_m=p_m,
            max_iter=max_iter,
            stop_fitness=stop_fitness,
            mutation_range=mutation_range,
        )
        self.beta = beta
        self.gamma_L = gamma_L
        self.gamma_U = gamma_U
        self.alpha = alpha

        self.gamma_t = gamma_U

    @override
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
        diff = np.abs(entropy_real - np.array(entropy_pred))

        if np.any(diff >= self.gamma_t):
            return 0
        return 1 / (1 + np.sum(np.abs((entropy_real - entropy_pred))))

    @override
    def run(self) -> None:
        """Run the genetic algorithm to calibrate the model."""
        start_time = time()
        new_population = self._init_population()
        i = 0
        fitness_values = np.zeros(self.pop_size)
        pop_size = self.pop_size
        population: NDArray | None = None

        while (
            i < self.max_iter
            and max(fitness_values) <= self.stop_fitness
            or pop_size < 3
        ):
            population = np.array(new_population)
            new_population = []

            population.round(decimals=self.beta)  # 4a

            for chr_id, chromosome in enumerate(population):
                multi_model = MultiDW(
                    self.num_of_simulations,
                    N=np.shape(self.y_real)[1],
                    d=chromosome[0],
                    mu=chromosome[1],
                    t=max(self.t),
                    topology=self.topology,
                    snapshots=self.t.tolist(),
                )
                self.abm_calls += self.num_of_simulations
                entropy_pred = multi_model.run()[-1]
                fitness_values[chr_id] = self._fitness(entropy_pred)

            fitness_values.round(decimals=self.beta)  # 4a

            zeros = np.where(fitness_values == 0)[0]
            fitness_values = np.delete(fitness_values, zeros)  # 4d
            population = np.delete(population, zeros, axis=0)  # 4d

            pop_size = np.shape(population)[0]

            if pop_size < 3:
                break

            for _ in range(int(pop_size * self.p_c // 2)):
                tournament = np.sort(np.random.choice(pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                weakest_index = np.argmin(f_vals)  # this is index of the weakest
                parents_ids = np.delete(tournament, weakest_index)
                parents = population[parents_ids]
                new_population.extend(self._crossover(parents))

            for _ in range(pop_size - int(pop_size * self.p_c // 2) * 2):
                tournament = np.sort(np.random.choice(pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                strongest_index = np.argmax(f_vals)
                strongest = population[tournament[strongest_index]]
                new_population.append(strongest)

            for chr_id in range(len(new_population)):
                if np.random.random() < self.p_m:
                    new_value_d = new_population[chr_id][0] + np.random.normal(
                        0, self.mutation_range
                    )
                    new_population[chr_id][0] = np.clip(
                        new_value_d, self.d_bounds[0], self.d_bounds[1]
                    )
                if np.random.random() < self.p_m:
                    new_value_mu = new_population[chr_id][1] + np.random.normal(
                        0, self.mutation_range
                    )
                    new_population[chr_id][1] = np.clip(
                        new_value_mu, self.mu_bounds[0], self.mu_bounds[1]
                    )

            i += 1
            self.gamma_t = self.gamma_L + (self.gamma_U - self.gamma_L) / (
                1 + i * self.alpha
            )
            if self.log:
                print(f"Iteration: {i}, Best fit: {max(fitness_values)}")
                print(f"Pop size: {pop_size}")

        assert population is not None
        self.best_fitness = np.max(fitness_values)
        self.best_params = population[np.argmax(fitness_values)]
        self.total_time = time() - start_time
        self.prediction_error = np.abs(
            np.array([self.real_d, self.real_mu]) - self.best_params
        )

    def export_calibration_results(self):
        """Return calibration results to csv file."""
        df = pd.DataFrame(
            {
                "d": [self.best_params[0]],
                "mu": [self.best_params[1]],
                "fitness": self.best_params,
                "prediction_error": self.prediction_error,
                "total_time": self.total_time,
                "abm_calls": self.abm_calls,
            }
        )
        df.to_csv(f"results/GA2_{self.name}.csv", index=False)
