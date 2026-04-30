from time import time
from typing import override

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from .model import Model
from .multiple_runs import MultiDW


class MLSurrogateCalibration(Model):
    """Machine Learning Surrogate Model for Calibration.

    This class uses machine learning models to build surrogate models for parameter calibration
    in multi-agent opinion dynamics simulations. It samples from a parameter pool, evaluates
    fitness using multiple simulations, and trains surrogate models to approximate the
    fitness landscape defined by the Deffuant-Weisbuch model.
    """

    def __init__(
        self,
        o_name: str,
        d_bounds: list[float],
        mu_bounds: list[float],
        pool_size: int,
        sample_size: int,
        max_iter: int,
        surrogate: str,
        real_d: float,
        real_mu: float,
        stop_fitness: float = 0.95,
        num_of_simulations: int = 50,
        topology: str = "full",
        log: bool = False,
    ):
        """Initialize the MLSurrogateCalibration instance.

        Parameters
        ----------
        o_name : str
            Name of the opinion file to read (without extension).
        pool_size : int
            Size of the parameter pool to generate.
        sample_size : int
            Number of samples to evaluate per iteration.
        max_iter : int
            Maximum number of iterations to run.
        num_of_simulations : int, optional
            Number of simulations to run for each parameter pair, by default 50.
        topology : str, optional
            Network topology for the simulations ('full', 'net', 'random', 'scale-free'), by default "full".
        log : bool, optional
            Whether to print iteration logs, by default False.
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
        )

        self.pool_size = pool_size
        self.sample_size = sample_size
        self.max_iter = min(max_iter, self.pool_size // self.sample_size)
        self.surrogate = surrogate
        self.stop_fitness = stop_fitness

        self.N = np.shape(self.y_real)[1]
        self.pool = self._init_pool()
        self.samples_indices = self._init_samples()  # pre defined samples indices
        self.x_train = []  # pairs of (d, mu) parameters for which the fitness was calculated
        self.y_train = []  # fitness values for the corresponding (d, mu) pairs in x_train
        self.fitness_estimation = []
        self.surrogate_model = self._init_surrogate_model(surrogate)

    @override
    def run(self):
        """Run the surrogate calibration process."""
        start_time = time()
        for i in range(self.max_iter):
            if self.log:
                print(f"Iteration {i + 1}/{self.max_iter}")
            samples = self.pool[self.samples_indices]
            self._calculate_fitness(samples)
            self.surrogate_model.fit(self.x_train, self.y_train)  # type: ignore
            # Predict fitness for the entire pool
            fitness_pred = self.surrogate_model.predict(self.pool)
            self.fitness_estimation.append(fitness_pred)
            # Update samples for the next iteration based on predicted fitness
            best_indices = np.argsort(fitness_pred)[-self.sample_size :]
            self.samples_indices = best_indices

            if np.max(fitness_pred) >= self.stop_fitness:
                if self.log:
                    print(
                        f"Stopping early at iteration {i + 1} with max fitness {np.max(fitness_pred):.4f}"
                    )
                break

        best_idx = np.argmax(self.y_train)
        self.best_params = np.array(self.x_train[best_idx])
        self.best_fitness = self.y_train[best_idx]
        self.total_time = time() - start_time
        self.prediction_error = np.abs(
            np.array([self.real_d, self.real_mu]) - self.best_params
        )

    def _init_pool(self):
        """Generate a large pool of parameters.

        Parameters
        ----------
        method : str
            Method of generating the pool (LHS or Sobol)
        """
        sampler = qmc.Sobol(2)
        pool = sampler.random(n=self.pool_size)
        return qmc.scale(
            pool,
            [self.d_bounds[0], self.mu_bounds[0]],
            [self.d_bounds[1], self.mu_bounds[1]],
        )

    def _init_samples(self):
        """Generate the sample from parameter pool."""
        return np.random.permutation(self.sample_size)  # .reshape(-1, self.sample_size)

    def _init_surrogate_model(self, surrogate: str):
        """Initialize the surrogate model based on the specified type.

        Parameters
        ----------
        surrogate : str
            Type of surrogate model to initialize ('GBR', 'RFR', 'GPR', 'MLP', 'XGB').

        Returns
        -------
        sklearn estimator
            Initialized surrogate model.
        """
        match surrogate:
            case "GBR":
                return GradientBoostingRegressor()
            case "RFR":
                return RandomForestRegressor()
            case "GPR":
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
                    noise_level=1
                )
                return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            case "MLP":
                return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
            case "XGB":
                return XGBRegressor(
                    objective="reg:squarederror", n_estimators=100, learning_rate=0.1
                )
            case _:
                raise ValueError("invalid model provided")

    def _calculate_fitness(self, samples):
        for s in samples:
            d, mu = s
            model = MultiDW(
                num_of_runs=self.num_of_simulations,
                N=self.N,
                d=d,
                mu=mu,
                t=max(self.t),
                topology=self.topology,
                snapshots=self.t.tolist(),
            )
            self.abm_calls += self.num_of_simulations
            entropy = model.run()[-1]
            self.x_train.append((d, mu))
            self.y_train.append(self._fitness(entropy))

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
        df.to_csv(f"results/MLSurrogate_{self.name}_{self.surrogate}.csv", index=False)
