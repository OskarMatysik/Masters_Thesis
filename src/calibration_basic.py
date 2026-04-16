import numpy as np
from multiple_runs import MultiDW
from scipy.stats import differential_entropy
import pandas as pd


class GridSearchCalibration:
    def __init__(self, o_name: str, d_bounds: list, mu_bounds: list, grid_size:int, num_of_simulations: int, 
                 topology: str = "full", log: bool = False):
        
        self.name = o_name
        self.t, self.y_real = self._read_opinions(o_name)
        self.d_bounds = d_bounds
        self.mu_bounds = mu_bounds
        self.grid_size = grid_size
        self.num_of_simulations = num_of_simulations
        self.topology = topology
        self.log = log
        self.d_grid, self.mu_grid = self._generate_grid()
        self.fitness_grid = np.zeros_like(self.d_grid)

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
        for i, (d, mu) in enumerate(zip(self.d_grid, self.mu_grid)):
            if self.log:
                print(f"Evaluating parameters: d={d:.4f}, mu={mu:.4f} ({i+1}/{len(self.d_grid)})")
            sim = MultiDW(N=self.y_real.shape[1], d=d, mu=mu, t=max(self.t), topology=self.topology, num_of_runs=self.num_of_simulations, snapshots=self.t)
            entropy = sim.run()[-1]
            fitness = self._fitness(entropy)
            self.fitness_grid[i] = fitness

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
    
class SimulatedAnnealingCalibration(GridSearchCalibration):
    def __init__(self, o_name: str, d_bounds: list, mu_bounds: list, initial_temp: float, cooling_rate: float, num_of_simulations: int, topology: str = "full", log: bool = False):
        super().__init__(o_name, d_bounds, mu_bounds, grid_size=0, num_of_simulations=num_of_simulations, topology=topology, log=log)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    
        
if __name__ == "__main__":
    cal = GridSearchCalibration(
        o_name = "o_N1000_d0.23_mu0.46_full",
        d_bounds = [0.1, 0.5],
        mu_bounds = [0.1, 0.5],
        grid_size = 15,
        num_of_simulations = 5,
        topology = "full",
        log = True 
        )
    cal.run()
