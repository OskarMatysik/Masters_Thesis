from models import DeffuantWeisbuchModel
from multiple_runs import MultiDW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ecdf, differential_entropy
from time import time

class GA1Calibration:
    """Genetic algorithm calibration using the Deffuant–Weisbuch model.

    This class performs GA-based parameter search by comparing model outcomes
    to real opinion data using entropy-based fitness evaluation.
    """
    def __init__(self, o_name: str, num_of_params:int, pop_size: int, p_c: float, p_m: float, 
                 max_iter: int, stop_fitness: float, L_p: float, U_p: float, mutation_range: float, 
                 topology: str, num_of_simulations: int, d_real: float, mu_real: float, log: bool = False) -> None:
        """Initialize the GA1 calibration.

        Args:
            o_name (str): Opinion data file name without extension.
            num_of_params (int): Number of genes in each chromosome.
            pop_size (int): Number of chromosomes in the population.
            p_c (float): Probability of performing crossover.
            p_m (float): Probability of mutating each gene.
            max_iter (int): Maximum number of algorithm iterations.
            stop_fitness (float): Fitness threshold for early stopping.
            L_p (Sequence[float]): Lower bounds for each parameter.
            U_p (Sequence[float]): Upper bounds for each parameter.
            mutation_range (float): Variance used to compute mutation standard deviation.
            topology (str): Network topology for the model simulations.
            num_of_simulations (int): Number of simulation runs used per chromosome.
            real_d (float): Real value for parameter d.
            real_mu (float): Real value for parameter mu.
            log (bool): If True, print progress updates during optimization.
        """
        self.name = o_name
        self.num_of_params = num_of_params
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.stop_fitness = stop_fitness
        self.L_p = L_p
        self.U_p = U_p
        self.mutation_range = np.sqrt(mutation_range) # standard deviation instead of variance
        self.topology = topology
        self.num_of_simulations = num_of_simulations
        self.d_real = d_real
        self.mu_real = mu_real
        self.log = log

        self.t, self.y_real = self._read_opinions(o_name)
        self.result = None
        self.fitness = None
        self.best = []
        self.t_converged = None

        self.total_time = None
        self.abm_calls = 0
        self.prediction_error = None


    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T
    

    def _fitness(self, entropy_pred: list) -> float:
        """
        Calculate fitness based on MSE between real and predicted CDFs for single simulation.
        y_pred (np.ndarray): Predicted opinions for the given time steps.
        
        1. ecdf approach - failed (add y_pred: np.ndarray = None to work)
        2. entropy calculation approach - good but slow (add y_pred: np.ndarray = None to work)
        3. entropy from multidw - current solution - entropy of y_pred instead of raw y_pred
        """

        # cdf_mse_values = []
        # x_eval = np.linspace(0, 1, 100)
        
        # for time_idx in range(len(self.t)):

        #     ecdf_real = ecdf(self.y_real[time_idx])
        #     ecdf_pred = ecdf(y_pred[time_idx])
            
        #     mse = np.sum((ecdf_real.cdf.evaluate(x_eval) - ecdf_pred.cdf.evaluate(x_eval)) ** 2)
        #     cdf_mse_values.append(mse)
        
        # total_mse = np.sum(cdf_mse_values)
        
        # return 1 / (1 + total_mse)
        

        entropy_real = np.array([differential_entropy(sample) for sample in self.y_real])
        entropy_pred = np.array(entropy_pred)
        # entropy_pred = np.array([differential_entropy(sample) for sample in y_pred])
        
        return 1 / (1 + np.sum(np.abs((entropy_real - entropy_pred))))
        
    
    def _init_population(self) -> np.ndarray:
        """Initialize the population for the genetic algorithm.
        
        Returns array with rows being chromosomes."""
        # population = np.empty(self.pop_size, self.num_of_params)
        # for _ in range(self.pop_size):
        #     d = np.random.uniform(self.L_p[0], self.U_p[0])
        #     mu = np.random.uniform(self.L_p[1], self.U_p[1])
        #     population.append((d, mu))
        parameters = []
        for _ in range(self.pop_size):
            parameters.append([np.random.uniform(self.L_p[i], self.U_p[i]) for i in range(self.num_of_params)])
        population = np.vstack(parameters)
        return population
    

    def _crossover(self, parents):
        """Perform crossover for 2 chosen chromosomes."""
        gene = np.random.choice(self.num_of_params)
        ratio = np.random.random() # for now U(0, 1), may change later
        chromosome_1, chromosome_2 = parents[0], parents[1]
        chromosome_1[gene] = chromosome_1[gene] * ratio + chromosome_2[gene] * (1 - ratio)
        chromosome_2[gene] = chromosome_2[gene] * ratio + chromosome_1[gene] * (1 - ratio)
        return chromosome_2, chromosome_2
    

    def run(self) -> None:
        """Run the genetic algorithm to calibrate the model."""
        start_time = time()
        new_population = self._init_population()
        i = 0
        fitness_values = np.zeros(self.pop_size)
        while i < self.max_iter and max(fitness_values) <= self.stop_fitness:
            
            population = np.array(new_population)
            new_population = []
            
            for chr_id, chromosome in enumerate(population):

                multi_model = MultiDW(self.num_of_simulations, N=np.shape(self.y_real)[1], d=chromosome[0], mu=chromosome[1], t=max(self.t), topology=self.topology, snapshots=self.t)
                self.abm_calls += self.num_of_simulations
                entropy_pred = multi_model.run()[-1]
                fitness_values[chr_id] = self._fitness(entropy_pred)

            for _ in range(int(self.pop_size * self.p_c // 2)):
                tournament = np.sort(np.random.choice(self.pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                weakest_index = np.argmin(f_vals) # this is index of the weakest
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
                for gene_id in range(self.num_of_params):
                    if np.random.random() < self.p_m:
                        new_value = new_population[chr_id][gene_id] + np.random.normal(0, self.mutation_range)
                        new_population[chr_id][gene_id] = np.clip(new_value, self.L_p[gene_id], self.U_p[gene_id])

            i += 1
            if self.log:
                print(f"Iteration: {i}, Best fit: {max(fitness_values)}")
            self.best.append(max(fitness_values))
        
        if i < self.max_iter:
            self.t_converged = i
        else:
            self.t_converged = self.max_iter
        
        self.fitness = fitness_values
        self.result = population
        self.total_time = time() - start_time
        self.prediction_error = np.abs(np.array([self.d_real, self.mu_real]) - self.result[np.argmax(self.fitness)])


    def export_final_params(self):
        """Export result to csv."""
        df = pd.DataFrame()
        for i in range(self.num_of_params):
            df[f"parameter_{i}"] = self.result[:, i]
        df["fitness"] = self.fitness
        df.sort_values(by=["fitness"], ascending=False)
        df.to_csv(f"results/GA1_calibration_{self.name}.csv", index=False)


    def export_calibration_results(self):
        """Return calibration results to csv file."""
        df = pd.DataFrame({
            "d": self.result[np.argmax(self.fitness), 0],
            "mu": self.result[np.argmax(self.fitness), 1],
            "fitness": max(self.fitness),
            "prediction_error": self.prediction_error,
            "total_time": self.total_time,
            "abm_calls": self.abm_calls,
        })
        df.to_csv(f"results/GA1_{self.name}.csv", index=False)
        

    def plot_best(self):
        """Plot the best fitness values over iterations and save to file."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best)
        plt.axhline(y=self.stop_fitness, color='r', linestyle='--', label='Stop Fitness')
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title(f"Best Fitness Over Iterations - {self.name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"results/GA1_calibration_{self.name}.png", dpi=300, bbox_inches='tight')
        plt.close()


    def calibration_results(self):
        """Print the best parameters and fitness after calibration."""
        best_idx = np.argmax(self.fitness)
        best_params = self.result[best_idx]
        best_fitness = self.fitness[best_idx]
        print(f"Best Parameters: {best_params}, Best Fitness: {best_fitness}")
        

class GA2Calibration:
    """Genetic algorithm calibration with dynamic thresholding and duplicate removal.

    This class extends the GA1-style calibration to include parameter rounding,
    adaptive gamma thresholding, and duplicate chromosome pruning.
    """
    def __init__(self, o_name: str, num_of_params:int, pop_size: int, p_c: float, p_m: float, max_iter: int,
                 stop_fitness: float, L_p: float, U_p: float, mutation_range: float, num_of_simulations: int,
                 topology: str, d_real: float, mu_real: float, beta: int, gamma_L: float, gamma_U: float, 
                 alpha: float, log: bool = False) -> None:
        """Initialize the GA2 calibration.

        Args:
            o_name (str): Opinion data file name without extension.
            num_of_params (int): Number of genes in each chromosome.
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
        self.name = o_name
        self.num_of_params = num_of_params
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.stop_fitness = stop_fitness
        self.L_p = L_p
        self.U_p = U_p
        self.mutation_range = np.sqrt(mutation_range) # standard deviation instead of variance
        self.num_of_simulations = num_of_simulations
        self.topology = topology
        self.beta = beta
        self.gamma_L = gamma_L
        self.gamma_U = gamma_U
        self.alpha = alpha 
        self.log = log
        self.d_real = d_real
        self.mu_real = mu_real

        self.t, self.y_real = self._read_opinions(o_name)
        self.result = None
        self.fitness = None
        self.best = []
        self.t_converged = None
        self.gamma_t = gamma_U

        self.total_time = None
        self.abm_calls = 0
        self.prediction_error = None
        
    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T
    
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
        diff = np.abs(entropy_real - entropy_pred)

        if np.any(diff >= self.gamma_t):
            return 0
        return 1 / (1 + np.sum(np.abs((entropy_real - entropy_pred))))
    
    def _init_population(self) -> np.ndarray:
        """Initialize the population for the genetic algorithm.
        Returns array with rows being chromosomes."""
        parameters = []
        for _ in range(self.pop_size):
            parameters.append([np.random.uniform(self.L_p[i], self.U_p[i]) for i in range(self.num_of_params)])
        population = np.vstack(parameters)
        return population

    def _crossover(self, parents):
        """Perform crossover for 2 chosen chromosomes."""
        gene = np.random.choice(self.num_of_params)
        ratio = np.random.random() # for now U(0, 1), may change later
        chromosome_1, chromosome_2 = parents[0], parents[1]
        chromosome_1[gene] = chromosome_1[gene] * ratio + chromosome_2[gene] * (1 - ratio)
        chromosome_2[gene] = chromosome_2[gene] * ratio + chromosome_1[gene] * (1 - ratio)
        return chromosome_2, chromosome_2
    
    def run(self) -> None:
        """Run the genetic algorithm to calibrate the model."""
        start_time = time()
        new_population = self._init_population()
        i = 0
        fitness_values = np.zeros(self.pop_size)
        pop_size = self.pop_size
        while i < self.max_iter and max(fitness_values) <= self.stop_fitness or pop_size < 3:
            
            population = np.array(new_population)
            new_population = []

            population.round(decimals=self.beta) # 4a
            
            for chr_id, chromosome in enumerate(population):

                multi_model = MultiDW(self.num_of_simulations, N=np.shape(self.y_real)[1], d=chromosome[0], mu=chromosome[1], t=max(self.t), topology=self.topology, snapshots=self.t)
                self.abm_calls += self.num_of_simulations
                entropy_pred = multi_model.run()[-1]
                fitness_values[chr_id] = self._fitness(entropy_pred)
            
            fitness_values.round(decimals=self.beta) # 4a
            
            zeros = np.where(fitness_values == 0)[0]
            fitness_values = np.delete(fitness_values, zeros) # 4d
            population = np.delete(population, zeros, axis=0) # 4d
            # population, unique = np.unique(population, axis=0, return_index=True) # 4b
            # fitness_values = fitness_values[unique] # 4b
            
            pop_size = np.shape(population)[0]

            if pop_size < 3:
                break
                
            for _ in range(int(pop_size * self.p_c // 2)):
                tournament = np.sort(np.random.choice(pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                weakest_index = np.argmin(f_vals) # this is index of the weakest
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
                for gene_id in range(self.num_of_params):
                    if np.random.random() < self.p_m:
                        new_value = new_population[chr_id][gene_id] + np.random.normal(0, self.mutation_range)
                        new_population[chr_id][gene_id] = np.clip(new_value, self.L_p[gene_id], self.U_p[gene_id])

            i += 1
            self.gamma_t = self.gamma_L + (self.gamma_U - self.gamma_L)/(1 + i * self.alpha)
            if self.log:
                print(f"Iteration: {i}, Best fit: {max(fitness_values)}")
                print(f"Pop size: {pop_size}")
            self.best.append(max(fitness_values))
        
        if i < self.max_iter:
            self.t_converged = i
        else:
            self.t_converged = self.max_iter
        
        self.fitness = fitness_values
        self.result = population
        self.total_time = time() - start_time
        self.prediction_error = np.abs(np.array([self.d_real, self.mu_real]) - self.result[np.argmax(self.fitness)])


    def export_final_params(self):
        """Export result to csv."""
        df = pd.DataFrame()
        for i in range(self.num_of_params):
            df[f"parameter_{i}"] = self.result[:, i]
        df["fitness"] = self.fitness
        df.sort_values(by=["fitness"], ascending=False)
        df.to_csv(f"results/GA2_calibration_{self.name}.csv", index=False)
        

    def export_calibration_results(self):
        """Return calibration results to csv file."""
        df = pd.DataFrame({
            "d": self.result[np.argmax(self.fitness), 0],
            "mu": self.result[np.argmax(self.fitness), 1],
            "fitness": max(self.fitness),
            "prediction_error": self.prediction_error,
            "total_time": self.total_time,
            "abm_calls": self.abm_calls,
        })
        df.to_csv(f"results/GA2_{self.name}.csv", index=False)
        
    def plot_best(self):
        """Plot the best fitness values over iterations and save to file."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best)
        plt.axhline(y=self.stop_fitness, color='r', linestyle='--', label='Stop Fitness')
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title(f"Best Fitness Over Iterations - {self.name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"results/GA2_calibration_{self.name}.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    calibration = GA1Calibration(
            o_name="o_N1000_d0.23_mu0.46_full", 
            num_of_params=2, 
            pop_size=50, 
            p_c=0.7,
            p_m=0.1, 
            max_iter=2, 
            stop_fitness=0.95, 
            L_p=[0, 0], 
            U_p=[0.5, 0.5], 
            mutation_range=0.005, 
            topology="full", 
            num_of_simulations=1,
            real_d=0.23,
            real_mu=0.46,
            log=True
        )
    calibration_2 = GA2Calibration(
            o_name="o_N1000_d0.23_mu0.46_full", 
            num_of_params=2, 
            pop_size=50, 
            p_c=0.7,
            p_m=0.1, 
            max_iter=2, 
            stop_fitness=0.95, 
            L_p=[0, 0], 
            U_p=[0.5, 0.5], 
            mutation_range=0.005, 
            num_of_simulations=1,           
            topology="full", 
            beta=6,
            gamma_L=2,
            gamma_U=10,
            alpha=0.2,
            real_d=0.23,
            real_mu=0.46,
            log=True
        )
    calibration.run()
    calibration.export_calibration_results()
    calibration_2.run()
    calibration_2.export_calibration_results()
