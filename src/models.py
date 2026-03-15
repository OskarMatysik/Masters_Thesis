import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import differential_entropy

class DeffuantWeisbuchModel:
    def __init__(self, N: int, d:float, mu:float, t:int | None = None) -> None:
        """Parameters:
        N: Number of agents
        d: Disagreement threshold
        mu: Confidence level
        t: Number of time steps
        """
        self.N = N
        self.d = d
        self.mu = mu
        self.t = t
        self.x = np.random.random(N)
        self.history = []
        self.converged = False

    def run(self) -> None:
        """Run the model for t time steps."""
        if self.t is None:
            while not self.converged:
                sorted = np.sort(self.x)
                distances = (np.roll(sorted, -1) - sorted)[:-1]
                if not np.any(np.logical_and(distances <= self.d, distances >= 1e-2)) and len(self.history) > 20:
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
        for _ in range(self.N):
            i, j = np.random.choice(self.N, 2, replace=False)
            if np.abs(self.x[i] - self.x[j]) < self.d:
                self.x[i] += self.mu * (self.x[j] - self.x[i])
                self.x[j] += self.mu * (self.x[i] - self.x[j])
                
    def _clusters(self) -> tuple[int, list]:
        """Calculate clusters of opinions. Return number of clusters and size of each cluster."""
        sorted_opinions = np.sort(self.x)
        cluster_count = 1
        sizes = [1]
        for i in range(1, len(sorted_opinions)):
            if sorted_opinions[i] - sorted_opinions[i - 1] < self.d / 2:
                sizes[-1] += 1
            else:
                cluster_count += 1
                sizes.append(1)
        return cluster_count, sizes

    def statistics(self) -> tuple[np.float64, int, list, np.float64]:
        """Calculate statistics of the final opinions."""
        std = np.float64(np.std(self.x))
        cluster_count, cluster_sizes = self._clusters()
        entropy = np.float64(differential_entropy(self.x, method="vasicek"))
        return std, cluster_count, cluster_sizes, entropy

    def plot_time_chart(self) -> None:
        """Plot the time chart of opinions."""
        plt.figure(figsize=(10, 6))
        plt.scatter(np.broadcast_to(np.arange(self.t), (self.N, self.t)).T, self.history, alpha=0.1, color="black", s=1, label="Opinions")
        plt.xlabel("Time Steps")
        plt.ylabel("Opinion")
        plt.title("Deffuant-Weisbuch Model Time Chart")
        plt.legend()
        plt.savefig(f"tests/deffuant_weisbuch/N{self.N}_d{self.d}_mu{self.mu}_time_chart.png")

    def plot_final_vs_initial(self) -> None:
        """Plot the final opinions vs initial opinions."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.history[0], self.history[-1], alpha=0.5, color="blue", s=10)
        plt.xlabel("Initial Opinion")
        plt.ylabel("Final Opinion")
        plt.title("Deffuant-Weisbuch Model: Final vs Initial Opinions")
        plt.savefig(f"tests/deffuant_weisbuch/N{self.N}_d{self.d}_mu{self.mu}_final_vs_initial.png")
        

if __name__ == "__main__":
    pass