from models import DeffuantWeisbuchModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ecdf, differential_entropy

class GA1Calibration:
    def __init__(self, o_name: str, num_of_params:int, pop_size: int, p_c: float, p_m: float, max_iter: int, stop_fitness: float, L_p: float, U_p: float, mutation_range: float, topology: str, log: bool = False) -> None:
        """Parameters: 
        o_name (str): Name of the opinion file
        num_of_params: Number of parameters (genes) of the model
        pop_size (int): Size of the population for the genetic algorithm
        p_c (float): Crossover probability
        p_m (float): Mutation probability
        max_iter (int): Maximum number of iterations for the genetic algorithm
        stop_fitness (float): Fitness value to stop the algorithm
        L_p (float): Lower bound for the parameters
        U_p (float): Upper bound for the parameters"""
        self.name = o_name
        self.t, self.y_real = self._read_opinions(o_name)
        self.num_of_params = num_of_params
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.stop_fitness = stop_fitness
        self.L_p = L_p
        self.U_p = U_p
        self.topology = topology
        self.mutation_range = np.sqrt(mutation_range) # standard deviation instead of variance
        self.log = log
        self.result = None
        self.fitness = None
        self.best = []
        self.t_converged = None

    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T
    
    def _fitness(self, y_pred: np.ndarray) -> float:
        """Calculate fitness based on MSE between real and predicted CDFs.
        y_pred (np.ndarray): Predicted opinions for the given time steps."""

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
        entropy_pred = np.array([differential_entropy(sample) for sample in y_pred])
        
        return 1 / (1 + np.sum(np.abs((entropy_real - entropy_pred)/entropy_real)))
        
    
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
        new_population = self._init_population()
        i = 0
        fitness_values = np.zeros(self.pop_size)
        while i < self.max_iter and max(fitness_values) <= self.stop_fitness:
            
            population = np.array(new_population)
            new_population = []
            
            for chr_id, chromosome in enumerate(population):
                model = DeffuantWeisbuchModel(N=np.shape(self.y_real)[1], d=chromosome[0], mu=chromosome[1], t=max(self.t), topology=self.topology)
                model.run()
                y_pred = np.sort(np.array([model.history[t-1] for t in self.t]), axis=1)
                fitness = self._fitness(y_pred)
                fitness_values[chr_id] = fitness
                
            for _ in range(int(self.pop_size * self.p_c // 2)):
                tournament = np.sort(np.random.choice(self.pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                weakest_index = np.argmin(f_vals) # this is index of the weakest
                parents_ids = np.delete(tournament, weakest_index)
                parents = population[parents_ids]
                new_population.extend(self._crossover(parents))
            
            for _ in range(self.pop_size - int(self.pop_size * self.p_c)):
                tournament = np.sort(np.random.choice(self.pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                strongest_index = np.argmax(f_vals)
                strongest = population[tournament[strongest_index]]
                new_population.append(strongest)
                
            for chr_id in range(self.pop_size):
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


    def export_result(self):
        """Export result to csv."""
        df = pd.DataFrame()
        for i in range(self.num_of_params):
            df[f"parameter_{i}"] = self.result[:, i]
        df["fitness"] = self.fitness
        df.sort_values(by=["fitness"], ascending=False)
        df.to_csv(f"results/GA1_calibration_{self.name}.csv")
        
        
    def plot_best(self):
        """Plot the best fitness values over iterations and save to file."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best)
        plt.axhline(y=0.75, color='r', linestyle='--', label='Stop Fitness')
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title(f"Best Fitness Over Iterations - {self.name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"results/GA1_calibration_{self.name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _loss_test(self):
        num_trajectories = 10
        trajectories = []
        
        # Generate 10 trajectories
        for _ in range(num_trajectories//2):
            model = DeffuantWeisbuchModel(N=np.shape(self.y_real)[1], d=0.23, mu=0.46, t=max(self.t), topology=self.topology)
            model.run()
            y_pred = np.sort(np.array([model.history[t] for t in self.t]), axis=1)
            trajectories.append(y_pred)
        
        for _ in range(num_trajectories//2):
            model = DeffuantWeisbuchModel(N=np.shape(self.y_real)[1], d=0.13, mu=0.36, t=max(self.t), topology=self.topology)
            model.run()
            y_pred = np.sort(np.array([model.history[t] for t in self.t]), axis=1)
            trajectories.append(y_pred)
        
        # Create 5 subplots comparing CDF of trajectories with y_real
        num_subplots = min(5, len(self.t))
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3*num_subplots))
        
        if num_subplots == 1:
            axes = [axes]
        
        # Create x values for CDF evaluation
        x_eval = np.linspace(0, 1, 200)
        
        for idx in range(num_subplots):
            time_idx = idx * (len(self.t) - 1) // max(1, num_subplots - 1) if num_subplots > 1 else idx
            
            # Plot CDF for all trajectories
            for traj_id, y_pred in enumerate(trajectories):
                # Compute empirical CDF
                sorted_data = np.sort(y_pred[time_idx])
                cdf_values = np.searchsorted(sorted_data, x_eval, side='right') / len(sorted_data)
                if traj_id == 0:
                    axes[idx].plot(x_eval, cdf_values, alpha=0.3, linewidth=1, label='CDF (trajectories)')
                else:
                    axes[idx].plot(x_eval, cdf_values, alpha=0.3, linewidth=1)
            
            # Plot CDF for real data
            sorted_real = np.sort(self.y_real[time_idx])
            cdf_real = np.searchsorted(sorted_real, x_eval, side='right') / len(sorted_real)
            axes[idx].plot(x_eval, cdf_real, label='CDF (real)', alpha=0.8, linewidth=2.5, color='red')
            
            axes[idx].set_xlabel("Opinion")
            axes[idx].set_ylabel("Cumulative Probability")
            axes[idx].set_title(f"Opinion Distribution (CDF) at t={self.t[time_idx]}")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"results/GA1_loss_test_{self.name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        

class GA2Calibration:
    def __init__(self, o_name: str, num_of_params:int, pop_size: int, p_c: float, p_m: float, max_iter: int,
                 stop_fitness: float, L_p: float, U_p: float, mutation_range: float, topology: str,
                 beta: int, gamma_L: float, gamma_U: float, alpha: float, log: bool = False) -> None:
        """Parameters: 
        o_name (str): Name of the opinion file
        num_of_params: Number of parameters (genes) of the model
        pop_size (int): Size of the population for the genetic algorithm
        p_c (float): Crossover probability
        p_m (float): Mutation probability
        max_iter (int): Maximum number of iterations for the genetic algorithm
        stop_fitness (float): Fitness value to stop the algorithm
        L_p (float): Lower bound for the parameters
        U_p (float): Upper bound for the parameters"""
        self.name = o_name
        self.t, self.y_real = self._read_opinions(o_name)
        self.num_of_params = num_of_params
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.stop_fitness = stop_fitness
        self.L_p = L_p
        self.U_p = U_p
        self.topology = topology
        self.mutation_range = np.sqrt(mutation_range) # standard deviation instead of variance
        self.log = log
        self.result = None
        self.fitness = None
        self.best = []
        self.t_converged = None
        self.beta = beta
        self.gamma_L = gamma_L
        self.gamma_U = gamma_U
        self.gamma_t = gamma_U
        self.alpha = alpha           
        
    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T
    
    def _fitness(self, y_pred: np.ndarray) -> float:
        """Calculate fitness based on MSE between real and predicted CDFs.
        y_pred (np.ndarray): Predicted opinions for the given time steps."""
        # cdf_mse_values = []
        # x_eval = np.linspace(0, 1, 100)
        
        # for time_idx in range(len(self.t)):

        #     ecdf_real = ecdf(self.y_real[time_idx])
        #     ecdf_pred = ecdf(y_pred[time_idx])
            
        #     mse = np.sum((ecdf_real.cdf.evaluate(x_eval) - ecdf_pred.cdf.evaluate(x_eval)) ** 2)
        #     if mse >= self.gamma_t: # 4c
        #         return 0
        #     cdf_mse_values.append(mse)
        
        # total_mse = np.sum(cdf_mse_values)
        
        # return 1 / (1 + total_mse)
        entropy_real = np.array([differential_entropy(sample) for sample in self.y_real])
        entropy_pred = np.array([differential_entropy(sample) for sample in y_pred])
        
        return 1 / (1 + np.sum(np.abs(entropy_real - entropy_pred)))
    
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
        new_population = self._init_population()
        i = 0
        fitness_values = np.zeros(self.pop_size)
        while i < self.max_iter and max(fitness_values) <= self.stop_fitness:
            
            population = np.array(new_population)
            new_population = []

            population.round(decimals=self.beta) # 4a
            
            for chr_id, chromosome in enumerate(population):
                model = DeffuantWeisbuchModel(N=np.shape(self.y_real)[1], d=chromosome[0], mu=chromosome[1], t=max(self.t), topology=self.topology)
                model.run()
                y_pred = np.sort(np.array([model.history[t-1] for t in self.t]), axis=1)
                fitness = self._fitness(y_pred)
                fitness_values[chr_id] = fitness
            
            fitness_values.round(decimals=self.beta) # 4a
            
            zeros = np.where(fitness_values == 0)[0]
            fitness_values = np.delete(fitness_values, zeros) # 4d
            population = np.delete(population, zeros, axis=0) # 4d
            population, unique = np.unique(population, axis=0, return_index=True) # 4b
            fitness_values = fitness_values[unique] # 4b
            
            pop_size = np.shape(population)[0]
                
            for _ in range(int(pop_size * self.p_c // 2)):
                tournament = np.sort(np.random.choice(pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                weakest_index = np.argmin(f_vals) # this is index of the weakest
                parents_ids = np.delete(tournament, weakest_index)
                parents = population[parents_ids]
                new_population.extend(self._crossover(parents))
            
            for _ in range(pop_size - int(pop_size * self.p_c // 2 * 2)):
                tournament = np.sort(np.random.choice(pop_size, 3, replace=False))
                f_vals = fitness_values[tournament]
                strongest_index = np.argmax(f_vals)
                strongest = population[tournament[strongest_index]]
                new_population.append(strongest)
            
            print("Pop size", pop_size, "\n", "New pop size", len(new_population))
                
            for chr_id in range(len(new_population)):
                for gene_id in range(self.num_of_params):
                    if np.random.random() < self.p_m:
                        new_value = new_population[chr_id][gene_id] + np.random.normal(0, self.mutation_range)
                        new_population[chr_id][gene_id] = np.clip(new_value, self.L_p[gene_id], self.U_p[gene_id])

            i += 1
            self.gamma_t = self.gamma_L + (self.gamma_U - self.gamma_L)/(1 + i * self.alpha)
            if self.log:
                print(f"Iteration: {i}, Best fit: {max(fitness_values)}")
            self.best.append(max(fitness_values))
        
        if i < self.max_iter:
            self.t_converged = i
        else:
            self.t_converged = self.max_iter
        
        self.fitness = fitness_values
        self.result = population


    def export_result(self):
        """Export result to csv."""
        df = pd.DataFrame()
        for i in range(self.num_of_params):
            df[f"parameter_{i}"] = self.result[:, i]
        df["fitness"] = self.fitness
        df.sort_values(by=["fitness"], ascending=False)
        df.to_csv(f"results/GA2_calibration_{self.name}.csv")
        
        
    def plot_best(self):
        """Plot the best fitness values over iterations and save to file."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best)
        plt.axhline(y=0.75, color='r', linestyle='--', label='Stop Fitness')
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
            pop_size=100, 
            p_c=0.7,
            p_m=0.1, 
            max_iter=50, 
            stop_fitness=0.9, 
            L_p=[0, 0], 
            U_p=[0.5, 0.5], 
            mutation_range=0.01, 
            topology="full", 
            log=True
        )
    calibration_2 = GA2Calibration(
            o_name="o_N1000_d0.23_mu0.46_full", 
            num_of_params=2, 
            pop_size=100, 
            p_c=0.7,
            p_m=0.1, 
            max_iter=50, 
            stop_fitness=0.99, 
            L_p=[0, 0], 
            U_p=[0.5, 0.5], 
            mutation_range=0.01,            
            topology="full", 
            beta=3,
            gamma_L=10,
            gamma_U=30,
            alpha=0.2,
            log=True
        )
    calibration.run()
    calibration.export_result()
    calibration.plot_best()
    # calibration._loss_test()
    # calibration_2.run()
    # calibration_2.export_result()
    # calibration_2.plot_best()
    # calibration_2._loss_test()