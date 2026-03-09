import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

class DeffuantWeisbuchModel:
    def __init__(self, N: int, d:float, mu:float, t:int) -> None:
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

    def run(self) -> None:
        """Run the model for t time steps."""
        for _ in range(self.t):
            self._step()
            self.history.append(self.x.copy())

    def _step(self) -> None:
        """Perform one time step of the model."""
        for _ in range(self.N):
            i, j = np.random.choice(self.N, 2, replace=False)
            if np.abs(self.x[i] - self.x[j]) < self.d:
                self.x[i] += self.mu * (self.x[j] - self.x[i])
                self.x[j] += self.mu * (self.x[i] - self.x[j])

    def statistics(self) -> dict:
        """Calculate statistics of the final opinions."""
        pass

    def plot_time_chart(self) -> None:
        """Plot the time chart of opinions."""
        plt.figure(figsize=(10, 6))
        plt.scatter(np.broadcast_to(np.arange(self.t), (self.N, self.t)).T, self.history, alpha=0.25, color="black", s=1, label="Opinions")
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