#!/usr/bin/env -S uv run
"""idw_min_rmsd.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numba import njit
from numpy.typing import NDArray


def plot_data(ax: Axes) -> None:
    # Use the Axes object for plotting
    ax.plot([1, 2, 3], [4, 5, 6])


ocean_size: int = 390
num_intervals: int = 65
num_samples: int = 220


def act_height(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    # Calculate the height of the "actual" ocean at (x,y)
    return np.array(
        (
            30 * np.sin(y / 40) * np.cos(x / 40)
            + 50 * np.sin(np.sqrt(x * x + y * y) / 40)
        )
        - 800,
        dtype=np.float64,
    )


def init_samples():
    np.random.seed(2016)

    global ocean_size, num_intervals, num_samples
    ocean_size = 390
    num_intervals = 65
    num_samples = 220

    global grid_x, grid_y, grid_z
    grid_x, grid_y = np.mgrid[
        # See numpy.mgrid() docs for why using complex() for step length
        0 : ocean_size : complex(0, num_intervals),
        0 : ocean_size : complex(0, num_intervals),
    ]
    grid_z = act_height(grid_x, grid_y)

    global samples_x, samples_y, samples_z
    samples_x = np.random.uniform(0, ocean_size, num_samples)
    samples_y = np.random.uniform(0, ocean_size, num_samples)
    samples_z = act_height(samples_x, samples_y)


@njit
def calc_idw_height(xi: int, yi: int, p: float):
    sum_weight = 0.0
    sum_height_weight = 0.0
    for si in range(num_samples):
        distance = np.hypot(
            grid_x[xi, xi] - samples_x[si], grid_y[yi, yi] - samples_y[si]
        )
        if distance == 0:
            return float(samples_z[si])
        weight: float = 1.0 / np.power(distance, p)
        sum_weight += weight
        sum_height_weight += samples_z[si] * weight
    return sum_height_weight / sum_weight


def est_height(p: float) -> NDArray[np.float64]:
    global est_z
    est_z = np.zeros_like(grid_x)
    for xi in range(num_intervals):
        for yi in range(num_intervals):
            est_z[xi, yi] = calc_idw_height(xi, yi, p)
    return est_z


def calc_rmsd(p: float) -> NDArray[np.float64]:
    sum_errors = 0.0
    for xi in range(num_intervals):
        for yi in range(num_intervals):
            act: float = grid_z[xi, yi]
            est: float = calc_idw_height(xi, yi, p)
            sum_errors += (act - est) ** 2
    rmsd = np.sqrt(sum_errors / num_samples)
    return np.array(rmsd, dtype=np.float64)


def plot(ax: Axes):
    p: NDArray[np.float64] = np.linspace(1.0, 9.0, 50)
    calc_rmsd_vec = np.vectorize(calc_rmsd)
    rmsd = calc_rmsd_vec(p)

    min_rmsd: int = np.amin(rmsd)
    best_p: float = p[np.argmin(rmsd)]

    ax.plot(p, rmsd)
    ax.scatter(best_p, min_rmsd, color="red")

    ax.set_title("Inverse Distance Weighting (p vs RMSD)")
    ax.set_xlabel("p (Power) term")
    ax.set_ylabel("RMSD (Act vs. Est)")

    ax.text(5.0, 45.0, f"best p = {best_p:.4f}", ha="left")


def main():
    init_samples()

    plt.figure(Path(__file__).name)
    plot(plt.axes())
    plt.show()


if __name__ == "__main__":
    main()
