#!/usr/bin/env -S uv run
"""k_means_spectral.py"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons

warnings.filterwarnings("ignore")


def plot_kmeans(ax, x):
    ax.set_title("k-Means Clustering")
    k_means = KMeans(2, n_init="auto", random_state=0)
    labels = k_means.fit_predict(x)
    ax.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap="viridis")


def plot_spectral(ax, x):
    ax.set_title("Spectral Clustering")
    model = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", assign_labels="kmeans"
    )
    labels = model.fit_predict(x)
    ax.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap="viridis")


def main():
    x = make_moons(200, noise=0.05, random_state=0)[0]

    plt.figure(Path(__file__).name)

    plot_kmeans(plt.subplot(2, 1, 1), x)
    plot_spectral(plt.subplot(2, 1, 2), x)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
