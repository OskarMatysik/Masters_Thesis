from models import *
from multiprocessing import Pool

class MultiDeffuantWeisbuch:
    def __init__(self, num_of_runs: int,  N: int, d:float, mu:float, t:int | None = None, num_of_cores: int = 6) -> None:
        self.num_of_runs = num_of_runs
        self.num_of_cores = num_of_cores
        self.N = N
        self.d = d
        self.mu = mu
        self.t = t
        self.chunks = [np.arange(num_of_runs)[i::num_of_cores] for i in range(num_of_cores)]

    def run(self) -> None:
        """Run the model given number of times and save the results."""
        with Pool(self.num_of_cores) as pool:
            results = pool.map(self._mapper, [chunk for chunk in self.chunks])
        return self.statistics(results)
    
    def _mapper(self, chunk: np.ndarray) -> tuple[float, int, list, float]:
        """Run the model and return the statistics."""
        chunk_results = []
        for _ in chunk:
            model = DeffuantWeisbuchModel(N=self.N, d=self.d, mu=self.mu, t=self.t)
            model.run()
            chunk_results.append(model.statistics())
        return chunk_results
    
    def statistics(self, results) -> tuple[float, float, float, float]:
        """Calculate statistics of the results."""
        std = []
        num_of_clusters = []
        cluster_sizes = []
        entropy = []
        for chunk in results:
            for result in chunk:
                s, cc, cs, e = result
                std.append(s)
                num_of_clusters.append(cc)
                cluster_sizes.extend(cs)
                entropy.append(e)
        breakpoint()
        return np.mean(std), np.mean(num_of_clusters), np.mean(cluster_sizes), np.mean(entropy)
                
    
if __name__ == "__main__":
    multi_model = MultiDeffuantWeisbuch(num_of_runs=20, N=1000, d=0.35, mu=0.5, t=50)
    stats = multi_model.run()
    # breakpoint()


    