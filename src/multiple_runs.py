import os
from multiprocessing import Pool
from time import time

import matplotlib.pyplot as plt
import numpy as np

from .models import DeffuantWeisbuchModel


class MultiDW:
    def __init__(
        self,
        num_of_runs: int,
        N: int,
        d: float,
        mu: float,
        t: int,
        topology: str = "full",
        num_of_cores: int = 16,
        snapshots: list | None = None,
    ) -> None:
        self.num_of_runs = num_of_runs
        self.num_of_cores = num_of_cores
        self.N = N
        self.d = d
        self.mu = mu
        self.t = t
        self.topology = topology
        self.chunks = [
            np.arange(num_of_runs)[i::num_of_cores] for i in range(num_of_cores)
        ]
        self.snapshots = snapshots

    def run(self):
        """Run the model given number of times and save the results."""
        with Pool(self.num_of_cores) as pool:
            results = pool.map(self._mapper, [chunk for chunk in self.chunks])
        return self.statistics(results)

    def _mapper(
        self, chunk: np.ndarray
    ) -> list[tuple[list[float], list[float], list[float]]]:
        """Run the model and return the statistics."""
        np.random.seed(int.from_bytes(os.urandom(4), "big"))  # ai slop moment
        chunk_results = []
        for _ in chunk:
            model = DeffuantWeisbuchModel(
                N=self.N, d=self.d, mu=self.mu, t=self.t, topology=self.topology
            )
            model.run()
            chunk_results.append(model.statistics(self.snapshots))
        return chunk_results

    def statistics(self, results) -> tuple[list[float], list[float], list[float]]:
        """Calculate average statistics of the results.
        If snapshots is None return statistics for final opinions"""
        if self.snapshots is None:
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
            return (
                [np.mean(std).astype(float)],
                [np.mean(num_of_clusters).astype(float)],
                [np.mean(entropy).astype(float)],
            )
        else:
            std = [[] for _ in range(len(self.snapshots))]
            num_of_clusters = [[] for _ in range(len(self.snapshots))]
            cluster_sizes = [[] for _ in range(len(self.snapshots))]
            entropy = [[] for _ in range(len(self.snapshots))]
            for chunk in results:
                for result in chunk:
                    for i in range(len(self.snapshots)):
                        s, cc, cs, e = [result[stat_id][i] for stat_id in range(4)]
                        std[i].append(s)
                        num_of_clusters[i].append(cc)
                        cluster_sizes[i].append(cs)
                        entropy[i].append(e)

            return (
                [np.mean(s).astype(float) for s in std],
                [np.mean(cc).astype(float) for cc in num_of_clusters],
                [np.mean(e).astype(float) for e in entropy],
            )


class MultiDWWithParams:
    def __init__(
        self, num_of_runs: int, params: list, num_of_cores: int = 16, log: bool = False
    ) -> None:
        self.num_of_runs = num_of_runs
        self.num_of_cores = num_of_cores
        self.params = params
        self.avg_std = []
        self.avg_num_of_clusters = []
        self.avg_cluster_sizes = []
        self.avg_entropy = []
        self.log = log

    def run(self) -> None:
        """Run the model given number of times and save the results."""
        for N, d, mu, t, topology in self.params:
            start_time = time()
            multi_model = MultiDW(
                num_of_runs=self.num_of_runs,
                N=N,
                d=d,
                mu=mu,
                t=t,
                num_of_cores=self.num_of_cores,
                topology=topology,
            )
            std, num_of_clusters, entropy = multi_model.run()
            self.avg_std.append(std)
            self.avg_num_of_clusters.append(num_of_clusters)
            self.avg_entropy.append(entropy)
            if self.log:
                print(
                    f"Finished {N=}, {d=}, {mu=}, {t=}, {topology=} in {time() - start_time:.2f}s."
                )

    # The results path is always the same here.
    def plot_results(self) -> None:
        ds = [params[1] for params in self.params]
        """Plot the results."""
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(ds, self.avg_std, marker="o")
        plt.xlabel("d")
        plt.ylabel("Average Std Dev")
        plt.title("Average Std Dev vs d")

        plt.subplot(2, 2, 2)
        plt.plot(ds, self.avg_num_of_clusters, marker="o")
        plt.xlabel("d")
        plt.ylabel("Average Number of Clusters")
        plt.title("Average Number of Clusters vs d")

        plt.subplot(2, 2, 3)
        plt.plot(ds, self.avg_cluster_sizes, marker="o")
        plt.xlabel("d")
        plt.ylabel("Average Cluster Size")
        plt.title("Average Cluster Size vs d")

        plt.subplot(2, 2, 4)
        plt.plot(ds, self.avg_entropy, marker="o")
        plt.xlabel("d")
        plt.ylabel("Average Entropy")
        plt.title("Average Entropy vs d")

        plt.tight_layout()
        plt.savefig(
            f"multiple_simulations/multi_deffuant_weisbuch_{self.params[-1][-1]}/results.png"
        )


# This doesnt work for t = None
def generate_params(
    N: int, dl: float, dh: float, mu: float, t: int, topology: str, step: float
) -> list:
    """Generate parameters for MultiDWWithParams class."""
    ds = np.arange(dl, dh + step, step).astype(float)
    return [(N, d, mu, t, topology) for d in ds]


if __name__ == "__main__":
    # Testing on parameter ranges:
    # N = 1000
    # d = [0.05, 0.1, ..., 0.5]
    # mu = 0.5
    # t = 50
    params_full = generate_params(
        N=1000, dl=0.05, dh=0.5, mu=0.5, t=100, topology="full", step=0.01
    )
    params_random = generate_params(
        N=1000, dl=0.05, dh=0.5, mu=0.5, t=100, topology="random", step=0.01
    )
    params_scale_free = generate_params(
        N=1000, dl=0.05, dh=0.5, mu=0.5, t=100, topology="scale-free", step=0.01
    )
    params_net = generate_params(
        N=1000, dl=0.05, dh=0.5, mu=0.5, t=1000, topology="net", step=0.01
    )
    # params = params_full, params_random, params_scale_free#, params_net
    params = [params_full]
    for p in params:
        multi_model = MultiDWWithParams(num_of_runs=20, params=p, log=True)
        multi_model.run()
        multi_model.plot_results()
