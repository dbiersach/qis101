#!/usr/bin/env -S uv run
"""idw.py"""

import numpy as np
import pyvista as pv
from numba import njit

ocean_size: int = 390
num_intervals: int = 65
num_samples: int = 220


def act_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
            grid_x[xi, yi] - samples_x[si],
            grid_y[xi, yi] - samples_y[si],
        )
        if distance == 0:
            return float(samples_z[si])
        weight: float = 1.0 / np.power(distance, p)
        sum_weight += weight
        sum_height_weight += samples_z[si] * weight
    return sum_height_weight / sum_weight


def est_height(p: float) -> np.ndarray:
    global est_z
    est_z = np.zeros_like(grid_x)
    for xi in range(num_intervals):
        for yi in range(num_intervals):
            est_z[xi, yi] = calc_idw_height(xi, yi, p)
    return est_z


def make_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> pv.StructuredGrid:
    """Build a PyVista StructuredGrid from 2-D x/y/z arrays."""
    grid = pv.StructuredGrid(x, y, z)
    # Attach z as a scalar so colormaps track elevation
    grid["elevation"] = z.ravel(order="F")
    return grid


def plot(idw_plot_type: int):
    plotter = pv.Plotter()

    actual_surf = make_surface(grid_x, grid_y, grid_z)
    est_surf = make_surface(grid_x, grid_y, est_z)

    # Sample points cloud
    sample_pts = pv.PolyData(np.column_stack([samples_x, samples_y, samples_z]))

    if idw_plot_type == 1:
        plotter.add_mesh(
            actual_surf, scalars="elevation", cmap="Blues", show_scalar_bar=False
        )
        plotter.add_mesh(
            sample_pts, color="red", point_size=6, render_points_as_spheres=True
        )

    elif idw_plot_type == 2:
        plotter.add_mesh(
            est_surf, scalars="elevation", cmap="Blues", show_scalar_bar=False
        )
        plotter.add_mesh(
            sample_pts, color="red", point_size=6, render_points_as_spheres=True
        )

    elif idw_plot_type == 3:
        # Wireframe overlay: actual in blue, estimated in red
        plotter.add_mesh(
            actual_surf,
            scalars="elevation",
            cmap="Blues",
            style="wireframe",
            show_scalar_bar=False,
        )
        plotter.add_mesh(
            est_surf,
            scalars="elevation",
            cmap="Reds",
            style="wireframe",
            show_scalar_bar=False,
        )

    plotter.show()


def main():
    init_samples()
    # TODO: Adjust the p (power) value in the following line
    est_height(p=2.0)
    # TODO: Change the plot type (1,2,3) in the following line
    plot(idw_plot_type=1)


if __name__ == "__main__":
    main()
