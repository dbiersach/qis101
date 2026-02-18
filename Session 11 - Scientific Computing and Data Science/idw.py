#!/usr/bin/env -S uv run
"""idw.py"""

import mayavi.mlab as mlab
import numpy as np
from numba import njit
from numpy.typing import NDArray

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


def plot(idw_plot_type: int):
    if idw_plot_type == 1:
        mlab.surf(grid_x, grid_y, grid_z, colormap="ocean")

    if idw_plot_type == 2:
        mlab.surf(grid_x, grid_y, est_z, colormap="ocean")

    # fmt: off
    if idw_plot_type == 3:
        mlab.surf(grid_x, grid_y, grid_z,
                  colormap="Blues", representation="wireframe")

        mlab.surf(grid_x, grid_y, est_z,
                  colormap="Reds", representation="wireframe")

    if idw_plot_type == 1 or idw_plot_type == 2:
        mlab.points3d(samples_x, samples_y, samples_z,
                      scale_factor=3, color=(1, 0, 0))
    #fmt: off        

    mlab.show()


def main():
    init_samples()
    # TODO: Adjust the p (power) value in the following line
    est_height(p=2)
    # TODO: Change the plot type (1,2,3) in the following line
    plot(idw_plot_type=1)


if __name__ == "__main__":
    main()
