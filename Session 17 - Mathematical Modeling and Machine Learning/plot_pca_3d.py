#!/usr/bin/env -S uv run
"""plot_pca_3d.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA


def pdf(x):
    # fmt: off
    return np.array(
        (stats.norm(scale=0.25 / np.e).pdf(x) +
             stats.norm(scale=4 / np.e).pdf(x)
        ) * 0.5)
    # fmt: on


def main():
    np.random.seed(4)
    x = np.random.normal(scale=0.5, size=(30_000))
    y = np.random.normal(scale=0.5, size=(30_000))
    z = np.random.normal(scale=0.1, size=len(x))

    density = pdf(x) * pdf(y)
    pdf_z = pdf(5 * z)
    density *= pdf_z

    a = x + y
    b = 2 * y
    c = a - b + z
    norm = np.sqrt(a.var() + b.var())
    a /= norm
    b /= norm

    # Translates slice objects to concatenation along the second axis
    Y = np.c_[a, b, c]
    pca = PCA(n_components=3)
    pca.fit(Y)

    # Find principal axes using the PCA technique
    V = pca.components_.T
    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V

    x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]

    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)

    plt.figure(Path(__file__).name, figsize=(10, 8), constrained_layout=True)
    ax = plt.axes(projection="3d", elev=12, azim=-62)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Draw every 10th point
    ax.scatter(a[::10], b[::10], c[::10], c=density[::10], marker="+", alpha=0.4)

    # Draw the principal 2D plane that preserves
    # the maximum variance of the original 3D data
    # ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)

    plt.show()


if __name__ == "__main__":
    main()
