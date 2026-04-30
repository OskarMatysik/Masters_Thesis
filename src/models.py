import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import differential_entropy


class DeffuantWeisbuchModel:
    def __init__(
        self,
        N: int,
        d: float,
        mu: float,
        t: int,
        topology: str = "full",
        num_of_data_points: int | None = None,
    ) -> None:
        """Parameters:
        N: Number of agents
        d: Disagreement threshold
        mu: Confidence level
        t: Number of time steps
        topology: Topology of the network (full, random, scale-free, net)
        num_of_data_points: Number of data points to export to file (if None, export none)
        """
        if topology == "net":
            self.N = int(np.ceil(np.sqrt(N)) ** 2)
        else:
            self.N = N
        self.d = d
        self.mu = mu
        self.t = t
        self.x = np.random.random(self.N)
        self.history = []
        self.converged = False
        self.topology = topology
        self.neighborhood = self._generate_topology()
        self.num_of_data_points = num_of_data_points

    def run(self) -> None:
        """Run the model for t time steps."""
        self.history.append(self.x.copy())
        if self.t is None:
            while not self.converged:
                sorted = np.sort(self.x)
                distances = (np.roll(sorted, -1) - sorted)[:-1]
                if (
                    not np.any(np.logical_and(distances <= self.d, distances >= 1e-2))
                    and len(self.history) > 20
                ):
                    self.converged = True
                    self.t = len(self.history) + 1
                self._step()
                self.history.append(self.x.copy())
        else:
            for _ in range(self.t):
                self._step()
                self.history.append(self.x.copy())

    def _step(self) -> None:
        """Perform one time step of the model. Each step consists of N interactions."""
        for _ in range(self.N // 2):
            i = np.random.choice(self.N)
            j = np.random.choice(np.argwhere(self.neighborhood[i]).flatten())
            if np.abs(self.x[i] - self.x[j]) < self.d:
                self.x[i] += self.mu * (self.x[j] - self.x[i])
                self.x[j] += self.mu * (self.x[i] - self.x[j])

    def _generate_topology(self) -> np.ndarray:
        """Generate the topology of the network."""
        if self.topology == "full":
            return np.ones((self.N, self.N)) - np.eye(self.N)
        elif self.topology == "random":
            p = 0.1
            M = np.triu(np.random.rand(self.N, self.N) < p, 1)
            return M + M.T
        elif self.topology == "scale-free":
            G = nx.barabasi_albert_graph(self.N, int(np.sqrt(self.N)))
            return nx.to_numpy_array(G)
        elif self.topology == "net":
            M = np.zeros((self.N, self.N))
            size = np.sqrt(self.N).astype(int)
            for i in range(self.N):
                row = i // size
                col = i % size
                neighbors = np.array(
                    [
                        (col + c) % size + (row + r) % size * size
                        for r in range(-1, 2)
                        for c in range(-1, 2)
                    ]
                )
                M[i, neighbors] = 1

            return M - np.eye(self.N)
        else:
            raise ValueError("Invalid topology")

    def _clusters(self, t: int | None = None) -> tuple[int, list]:
        """Calculate clusters of opinions. Return number of clusters and size of each cluster.
        Two agents are considered in the same cluster if their opinions differ by less than d/2.
        If t is None returns cluster statistics of final opinions
        """
        if t is None:
            sorted_opinions = np.sort(self.x)
        else:
            sorted_opinions = np.sort(self.history[t])
        cluster_count = 1
        sizes = [1]
        for i in range(1, len(sorted_opinions)):
            if sorted_opinions[i] - sorted_opinions[i - 1] < self.d / 2:
                sizes[-1] += 1
            else:
                cluster_count += 1
                sizes.append(1)
        return cluster_count, sizes

    def statistics(self, snapshots: list | None = None):
        """Calculate statistics of the opinions. If t is None return statistics of final opinions."""
        if snapshots is None:
            std = float(np.std(self.x))
            cluster_count, cluster_sizes = self._clusters()
            entropy = float(differential_entropy(self.x, method="vasicek"))
        else:
            std = [float(np.std(self.history[t])) for t in snapshots]
            cluster_count = [self._clusters(t)[0] for t in snapshots]
            cluster_sizes = [self._clusters(t)[1] for t in snapshots]
            entropy = [
                float(differential_entropy(self.history[t], method="vasicek"))
                for t in snapshots
            ]

        return std, cluster_count, cluster_sizes, entropy

    def export_data(self) -> None:
        """Export opinions of agents at random time steps to a file."""
        if self.num_of_data_points is None:
            return
        indices = np.sort(
            np.random.choice(self.t, self.num_of_data_points, replace=False).astype(int)
        )
        data = pd.DataFrame(
            np.array([self.history[i] for i in indices]).T, columns=indices
        )
        data.to_csv(
            f"results/o_N{self.N}_d{self.d}_mu{self.mu}_{self.topology}.csv",
            index=False,
        )

    def plot_time_chart(self) -> None:
        """Plot the time chart of opinions."""
        plt.figure(figsize=(10, 6))
        plt.scatter(
            np.broadcast_to(np.arange(self.t + 1), (self.N, self.t + 1)).T,
            self.history,
            alpha=0.1,
            color="black",
            s=1,
            label="Opinions",
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Opinion")
        plt.title("Deffuant-Weisbuch Model Time Chart")
        plt.legend()
        plt.savefig(
            f"single_simulations/deffuant_weisbuch/{self.topology}/N{self.N}_d{self.d}_mu{self.mu}_time_chart.png"
        )

    def plot_final_vs_initial(self) -> None:
        """Plot the final opinions vs initial opinions."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.history[0], self.history[-1], alpha=0.5, color="blue", s=10)
        plt.xlabel("Initial Opinion")
        plt.ylabel("Final Opinion")
        plt.title("Deffuant-Weisbuch Model: Final vs Initial Opinions")
        plt.savefig(
            f"single_simulations/deffuant_weisbuch/{self.topology}/N{self.N}_d{self.d}_mu{self.mu}_final_init.png"
        )
