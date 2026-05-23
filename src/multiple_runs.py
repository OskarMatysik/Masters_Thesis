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
        np.random.seed(int.from_bytes(os.urandom(4), "big"))
        chunk_results = []
        for _ in chunk:
            model = DeffuantWeisbuchModel(
                N=self.N, d=self.d, mu=self.mu, t=self.t, topology=self.topology
            )
            model.run()
            chunk_results.append(model.statistics(self.snapshots))
        return chunk_results

    def statistics(self, results) -> tuple[list[float], list[float], list[list], list[list], list[float], list[np.ndarray]]:
        """Calculate average statistics of the results.
        If snapshots is None return statistics for final opinions"""
        if self.snapshots is None:
            std = []
            num_of_clusters = []
            cluster_sizes = []
            kdes = []
            hist = []
            entropy = []
            observations = []
            for chunk in results:
                for result in chunk:
                    s, cc, cs, kde, h, e, obs = result
                    std.append(s)
                    num_of_clusters.append(cc)
                    cluster_sizes.extend(cs)
                    kdes.append(kde)
                    hist.append(h)
                    entropy.append(e)
                    observations.append(obs)
            return (
                [np.mean(std).astype(float)],
                [np.mean(num_of_clusters).astype(float)],
                [np.mean(kdes, axis=0)],
                [np.mean(hist, axis=0)],
                [np.mean(entropy).astype(float)],
                [np.array(observations)],
            )
        else:
            std = [[] for _ in range(len(self.snapshots))]
            num_of_clusters = [[] for _ in range(len(self.snapshots))]
            cluster_sizes = [[] for _ in range(len(self.snapshots))]
            kdes = [[] for _ in range(len(self.snapshots))]
            hist = [[] for _ in range(len(self.snapshots))]
            entropy = [[] for _ in range(len(self.snapshots))]
            observations = [[] for _ in range(len(self.snapshots))]
            for chunk in results:
                for result in chunk:
                    for i in range(len(self.snapshots)):
                        s, cc, cs, kde, h, e, obs = [result[stat_id][i] for stat_id in range(7)]
                        std[i].append(s)
                        num_of_clusters[i].append(cc)
                        cluster_sizes[i].append(cs)
                        kdes[i].append(kde)
                        hist[i].append(h)
                        entropy[i].append(e)
                        observations[i].append(obs)
                        # breakpoint()

            return (
                [np.mean(s).astype(float) for s in std],
                [np.mean(cc).astype(float) for cc in num_of_clusters],
                [np.mean(kde, axis=0) for kde in kdes],
                [np.mean(h, axis=0) for h in hist],
                [np.mean(e).astype(float) for e in entropy],
                [np.array(obs) for obs in observations],
            )


