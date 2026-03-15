from models import *
from multiprocessing import Pool

class MultiDeffauntWeisbuch:
    def __init__(self, num_of_runs: int, N: int, d:float, mu:float, t:int) -> None:
        self.num_of_runs = num_of_runs
        self.N = N
        self.d = d
        self.mu = mu
        self.t = t

    def run(self) -> None:
        """Run the model given number of times and save the results."""
        with Pool(6) as pool:
            runs = pool.starmap(self.model.run, [(self.model,) for _ in range(self.num_of_runs)])
        raise NotImplementedError()


    