from multiple_runs import MultiDW
from scipy.stats import qmc
import numpy as np
import pandas as pd
from scipy.stats import differential_entropy

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


class MLSurrogateCalibration:
    """Machine Learning Surrogate Model for Calibration.
    
    This class uses machine learning models to build surrogate models for parameter calibration
    in multi-agent opinion dynamics simulations. It samples from a parameter pool, evaluates
    fitness using multiple simulations, and trains surrogate models to approximate the
    fitness landscape defined by the Deffuant-Weisbuch model.
    """

    def __init__(self, o_name: str, pool_size: int, sample_size: int, max_iter: int, 
                surrogate: str, sampling_method: str = "Sobol", stop_fitness: float = 0.95,
                 num_of_simulations: int = 50, topology: str = "full", log: bool = False):
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
        sampling_method : str, optional
            Method for generating the parameter pool ('Sobol' or 'LHS'), by default "Sobol".
        num_of_simulations : int, optional
            Number of simulations to run for each parameter pair, by default 50.
        topology : str, optional
            Network topology for the simulations ('full', 'net', 'random', 'scale-free'), by default "full".
        log : bool, optional
            Whether to print iteration logs, by default False.
        """
        self.o_name = o_name
        self.pool_size = pool_size
        self.sample_size = sample_size
        self.max_iter = min(max_iter, self.pool_size // self.sample_size)
        self.surrogate = surrogate
        self.sampling_method = sampling_method
        self.stop_fitness = stop_fitness
        self.num_of_simulations = num_of_simulations
        self.topology = topology
        self.log = log

        self.d_bounds = [0.001, 0.601]
        self.mu_bounds = [0.001, 0.501]
        self.t, self.y_real = self._read_opinions(o_name)
        self.N = np.shape(self.y_real)[1]
        self.pool = self._init_pool()
        self.samples_indices = self._init_samples() # pre defined samples indices
        self.x_train = [] # pairs of (d, mu) parameters for which the fitness was calculated
        self.y_train = [] # fitness values for the corresponding (d, mu) pairs in x_train
        self.fitness_estimation = []
        self.surrogate_model = self._init_surrogate_model(surrogate)
        


    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T


    def run(self):
        """Run the surrogate calibration process."""
        for i in range(self.max_iter):
            if self.log:
                print(f"Iteration {i+1}/{self.max_iter}")
            samples = self.pool[self.samples_indices]
            self._calculate_fitness(samples)
            self.surrogate_model.fit(self.x_train, self.y_train)
            # Predict fitness for the entire pool
            fitness_pred = self.surrogate_model.predict(self.pool)
            self.fitness_estimation.append(fitness_pred)
            # Update samples for the next iteration based on predicted fitness
            best_indices = np.argsort(fitness_pred)[-self.sample_size:]
            self.samples_indices = best_indices

            if np.max(fitness_pred) >= self.stop_fitness:
                if self.log:
                    print(f"Stopping early at iteration {i+1} with max fitness {np.max(fitness_pred):.4f}")
                break

            

    def _init_pool(self):
        """Generate a large pool of parameters.
        
        Parameters
        ----------
        method : str
            Method of generating the pool (LHS or Sobol)
        """
        match self.sampling_method:
            case "LHS":
                sampler = qmc.LatinHypercube(2)
            case "Sobol":
                sampler = qmc.Sobol(2)
        pool = sampler.random(n=self.pool_size)
        return qmc.scale(pool, [self.d_bounds[0], self.mu_bounds[0]], [self.d_bounds[1], self.mu_bounds[1]])      


    def _init_samples(self):
        """Generate the sample from parameter pool."""
        return np.random.permutation(self.sample_size)#.reshape(-1, self.sample_size)
    

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
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
                return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            case "MLP":
                return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
            case "XGB":
                return XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    

    def _calculate_fitness(self, samples):
        for s in samples:
            d, mu = s
            model = MultiDW(
                num_of_runs = self.num_of_simulations,
                N = self.N,
                d = d,
                mu = mu,
                t = max(self.t),
                topology = self.topology,
                snapshots = self.t
                )
            entropy = model.run()[-1]
            self.x_train.append((d, mu))
            self.y_train.append(self._fitness(entropy))


    def _fitness(self, entropy_pred: list) -> float:
        """
        Calculate fitness based on MSE between real and predicted CDFs for single simulation.
        y_pred (np.ndarray): Predicted opinions for the given time steps.
        
        1. ecdf approach - failed (add y_pred: np.ndarray = None to work)
        2. entropy calculation approach - good but slow (add y_pred: np.ndarray = None to work)
        3. entropy from multidw - current solution - entropy of y_pred instead of raw y_pred
        """

        entropy_real = np.array([differential_entropy(sample) for sample in self.y_real])
        entropy_pred = np.array(entropy_pred)
        
        return 1 / (1 + np.sum(np.abs(entropy_real - entropy_pred)))




if __name__ == "__main__":
    cal = MLSurrogateCalibration(
        o_name = "o_N1000_d0.23_mu0.46_full",
        pool_size = 1000,
        sample_size = 20,
        max_iter = 10,
        surrogate = "XGB",
        sampling_method = "Sobol",
        stop_fitness = 0.95,
        num_of_simulations = 5,
        topology = "full",
        log = True 
        )
    cal.run()
    breakpoint()

