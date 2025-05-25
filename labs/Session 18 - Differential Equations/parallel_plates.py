#!/usr/bin/env python3
"""parallel_plates.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import convolve, generate_binary_structure


def conductor_walls(a):
    # Carl Neumann (1832-1925)
    # A conductor forces the walls to
    # have zero potential (gradient=0)
    a[0, :] = a[1, :]
    a[-1, :] = a[-2, :]
    a[:, 0] = a[:, 1]
    a[:, -1] = a[:, -2]
    return a


def insulator_walls(a):
    # Johann Dirichlet (1805-1859)
    # An insulator forces the walls to
    # have a fixed charge (voltage=0)
    a[0, :] = 0
    a[-1, :] = 0
    a[:, 0] = 0
    a[:, -1] = 0
    return a


def solve_laplace(ax, boundary_func):
    left_volts = -1
    right_volts = 1

    N = 100
    grid = np.zeros((N, N))
    grid[30:70, 29:30] = left_volts
    grid[30:70, 70:71] = right_volts

    mask_neg = grid == left_volts
    mask_pos = grid == right_volts

    kern = generate_binary_structure(2, 1).astype(float) / 4
    kern[1, 1] = 0

    iters = 5000
    for _ in range(iters):
        # Average every four neighbor cells in the grid
        grid_next = convolve(grid, kern, mode="constant")
        # Reapply the boundary conditions
        grid_next = boundary_func(grid_next)
        # Reapply the plate voltages
        grid_next[mask_neg] = left_volts
        grid_next[mask_pos] = right_volts
        # The next grid becomes the current grid
        grid = grid_next

    # Render a colored contour plot of the electrostatic field potential
    surf = ax.contourf(range(N), range(N), grid, cmap="rainbow", levels=20)
    ax.get_figure().colorbar(surf, ax=ax, shrink=0.5, label="Volts")

    # Blacken the two parallel plates
    ax.add_patch(Rectangle((29, 30), 1, 40, edgecolor="k", facecolor="k"))
    ax.add_patch(Rectangle((70, 30), 1, 40, edgecolor="k", facecolor="k"))

    # Title each graph
    if boundary_func == conductor_walls:
        ax.set_title("Conductor Walls")
    else:
        ax.set_title("Insulator Walls")

    ax.get_figure().suptitle(
        "Electrostatic Field Potential\n"  # noqa
        "Surrounding a Parallel Plate Capacitor"
    )
    ax.set_aspect("equal")


def main():
    plt.figure(Path(__file__).name, figsize=(10, 5.5))
    solve_laplace(plt.subplot(1, 2, 1), conductor_walls)
    solve_laplace(plt.subplot(1, 2, 2), insulator_walls)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
