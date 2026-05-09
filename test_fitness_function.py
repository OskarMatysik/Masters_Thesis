from src.multiple_runs import MultiDW
import numpy as np
import matplotlib.pyplot as plt

def test_entropy_std():
    ds = np.linspace(0.05, 0.5, 10)
    mus = np.linspace(0.05, 0.5, 10)
    snapshots = [4, 24, 49]
    entropies = np.zeros((len(snapshots), len(ds), len(mus)))
    stds = np.zeros((len(snapshots), len(ds), len(mus)))
    clusters = np.zeros((len(snapshots), len(ds), len(mus)))
    for row, d in enumerate(ds):
        for col, mu in enumerate(mus):
            cal = MultiDW(
                N=1000,
                d=d,
                mu=mu,
                t=int(10 * (3 + 1 / mu)),
                topology="full",
                num_of_runs=20,
                snapshots=snapshots,
            )
            stats = cal.run()
            std, cluster, entropy = stats
            entropies[:, row, col] = entropy
            stds[:, row, col] = std
            clusters[:, row, col] = cluster
            print(f"Completed d={d}, mu={mu}")
    
    for i, snapshot in enumerate(snapshots):
        plt.figure(figsize=(10, 6))
        plt.imshow(entropies[i], extent=(0.05, 0.5, 0.05, 0.5), origin="lower", aspect="auto")
        plt.colorbar(label="Entropy")
        plt.title(f"Entropy at snapshot {snapshot}")
        plt.xlabel("d")
        plt.ylabel("mu")
        plt.savefig(f"results/entropy_snapshot_{snapshot}.png")

        plt.figure(figsize=(10, 6))
        plt.imshow(stds[i], extent=(0.05, 0.5, 0.05, 0.5), origin="lower", aspect="auto")
        plt.colorbar(label="Standard Deviation")
        plt.title(f"Standard Deviation at snapshot {snapshot}")
        plt.xlabel("d")
        plt.ylabel("mu")
        plt.savefig(f"results/std_snapshot_{snapshot}.png")

        plt.figure(figsize=(10, 6))
        plt.imshow(clusters[i], extent=(0.05, 0.5, 0.05, 0.5), origin="lower", aspect="auto")
        plt.colorbar(label="Number of Clusters")
        plt.title(f"Number of Clusters at snapshot {snapshot}")
        plt.xlabel("d")
        plt.ylabel("mu")
        plt.savefig(f"results/clusters_snapshot_{snapshot}.png")

if __name__ == "__main__":
    test_entropy_std()