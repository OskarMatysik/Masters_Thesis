from models import DeffuantWeisbuchModel
import numpy as np
import pandas as pd

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
        self.result = None
        self.fitness = None
        self.topology = topology
        self.mutation_range = np.sqrt(mutation_range) # standard deviation instead of variance
        self.log = log

    def _read_opinions(self, o_name: str) -> np.ndarray:
        """Read the opinions from the file.
        o_name (str): Name of the opinion file."""
        df = pd.read_csv(f"results/{o_name}.csv", header=0)
        t = np.array(df.columns, dtype=int)
        return t, np.sort(df.to_numpy(), axis=0).T
    
    def _loss(self, y_pred: np.ndarray) -> float:
        """Calculate the loss between the predicted and real opinions.
        y_pred (np.ndarray): Predicted opinions for the given time steps."""
        return np.mean((self.y_real - y_pred) ** 2, axis=0)

    def _fitness(self, loss: float) -> float:
        """Calculate the fitness from the loss.
        loss (float): Loss value for given time steps."""
        return 1 / (1 + np.sum(loss))
    
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
        population = self._init_population()
        i = 0
        fitness_values = np.zeros(self.pop_size)
        while i < self.max_iter and max(fitness_values) <= self.stop_fitness:
            
            new_population = []
            
            for chr_id, chromosome in enumerate(population):
                model = DeffuantWeisbuchModel(N=np.shape(self.y_real)[1], d=chromosome[0], mu=chromosome[1], t=max(self.t), topology=self.topology)
                model.run()
                y_pred = np.sort(np.array([model.history[t-1] for t in self.t]), axis=1)
                loss = self._loss(y_pred)
                fitness = self._fitness(loss)
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
            population = np.array(new_population)
            i += 1
            if self.log:
                print(f"Iteration: {i}, Best fit: {max(fitness_values)}")
        self.result = population

    
if __name__ == "__main__":
    calibration = GA1Calibration(
            o_name="o_N1000_d0.23_mu0.46_full", 
            num_of_params=2, 
            pop_size=100, 
            p_c=0.7,
            p_m=0.1, 
            max_iter=50, 
            stop_fitness=0.75, 
            L_p=[0, 0], 
            U_p=[0.5, 0.5], 
            mutation_range=0.01, 
            topology="full", 
            log=True
        )
    calibration.run()
    breakpoint()