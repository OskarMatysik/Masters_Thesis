from .models import DeffuantWeisbuchModel
import numpy as np
import pandas as pd

class GA1Calibration:
    def __init__(self, o_name: str, pop_size: int, p_c: float, p_m: float, max_iter: int, stop_fitness: float, L_p: float, U_p: float) -> None:
        """Parameters: 
        o_name (str): Name of the opinion file
        pop_size (int): Size of the population for the genetic algorithm
        p_c (float): Crossover probability
        p_m (float): Mutation probability
        max_iter (int): Maximum number of iterations for the genetic algorithm
        stop_fitness (float): Fitness value to stop the algorithm
        L_p (float): Lower bound for the parameters
        U_p (float): Upper bound for the parameters"""
        self.t, self.y_real = self._read_opinions(o_name)
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.stop_fitness = stop_fitness
        self.L_p = L_p
        self.U_p = U_p

    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, df.to_numpy().T
    
    def _loss(self, y_pred: np.ndarray) -> float:
        """Calculate the loss between the predicted and real opinions.
        y_pred (np.ndarray): Predicted opinions for the given time steps."""
        return np.mean((self.y_real - y_pred) ** 2, axis=0)

    def _fitness(self, loss: float) -> float:
        """Calculate the fitness from the loss.
        loss (float): Loss value for given time steps."""
        return 1 / (1 + np.sum(loss))
    
    def _init_population(self) -> list:
        """Initialize the population for the genetic algorithm."""
        population = []
        for _ in range(self.pop_size):
            d = np.random.uniform(self.L_p[0], self.U_p[0])
            mu = np.random.uniform(self.L_p[1], self.U_p[1])
            topology = np.random.choice(["full", "random", "scale-free", "net"])
            population.append((d, mu, topology))
        return population
    
    def _crossover(self, parents):
        """Perform crossover for 2 chosen chromosomes."""
        pass
    
    def run(self) -> None:
        """Run the genetic algorithm to calibrate the model."""
        population = self._init_population()
        i = 0
        fitness_values = [0] * self.pop_size
        new_population = []
        while i < self.max_iter and max(fitness_values) <= self.stop_fitness:
            for chr_id, chromosome in enumerate(population):
                N, d, mu, t, topology = chromosome
                model = DeffuantWeisbuchModel(N=N, d=d, mu=mu, t=t, topology=topology)
                model.run()
                y_pred = np.array(model.history[self.t])
                loss = self._loss(y_pred)
                fitness = self._fitness(loss)
                fitness_values[chr_id] = fitness
            for j in range(self.pop_size * self.p_c // 2):
                tournament = np.random.choice(self.pop_size, 3, replace=False)
                parents = sorted(fitness[tournament])[1:]
                new_population.extend(self._crossover(parents))
            for j in range(self.pop_size * (1 - self.p_c)):
                tournament = np.random.choice(self.pop_size, 3, replace=False)
                best = sorted(fitness[tournament])[-1]
                new_population.append(best)
            for chr_id in range(self.pop_size):
                for gene_id in range(len(population[0])):
                    if np.random.random() < self.p_m:
                        new_population[chr_id][gene_id] = np.random.uniform(self.L_p[gene_id], self.U_p[gene_id])
            population = new_population
            i += 1

    
if __name__ == "__main__":
    pass
        