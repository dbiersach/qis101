#!/usr/bin/env -S uv run
"""parallel_plates.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import convolve, generate_binary_structure


def zero_normal_field_walls(a):
    # Carl Neumann (1832-1925)
    # Homogeneous Neumann condition: no normal electric field.
    a[0, :] = a[1, :]
    a[-1, :] = a[-2, :]
    a[:, 0] = a[:, 1]
    a[:, -1] = a[:, -2]
    return a


def grounded_walls(a):
    # Johann Dirichlet (1805-1859)
    # Homogeneous Dirichlet condition: grounded walls have zero potential.
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

    mask_neg = np.zeros_like(grid, dtype=bool)
    mask_pos = np.zeros_like(grid, dtype=bool)
    mask_neg[30:70, 29:30] = True
    mask_pos[30:70, 70:71] = True

    kern = generate_binary_structure(2, 1).astype(float) / 4
    kern[1, 1] = 0

    max_iters = 100000
    tolerance = 1e-10  # maximum voltage change per Jacobi iteration
    for iteration in range(1, max_iters + 1):
        # Average every four neighbor cells in the grid
        grid_next = convolve(grid, kern, mode="constant")
        # Reapply the boundary conditions
        grid_next = boundary_func(grid_next)
        # Reapply the plate voltages
        grid_next[mask_neg] = left_volts
        grid_next[mask_pos] = right_volts
        # Check every grid point for a change in voltage
        max_change = np.max(np.abs(grid_next - grid))
        grid = grid_next
        if max_change < tolerance:
            break
    else:
        raise RuntimeError("Jacobi iteration did not converge")
    print(
        f"{boundary_func.__name__}: {iteration:,} iterations; "
        f"maximum update {max_change:.3e} V"
    )

    # Render a colored contour plot of the electrostatic field potential
    voltage_min = min(left_volts, right_volts, float(grid.min()))
    voltage_max = max(left_volts, right_volts, float(grid.max()))
    if voltage_max - voltage_min < 1e-5:
        voltage_min, voltage_max = left_volts - 1, left_volts + 1
    levels = np.linspace(voltage_min, voltage_max, 21)
    surf = ax.contourf(range(N), range(N), grid, cmap="rainbow", levels=levels)
    ax.get_figure().colorbar(surf, ax=ax, shrink=0.5, label="Volts")

    # Blacken the two parallel plates
    ax.add_patch(Rectangle((29, 30), 1, 40, edgecolor="k", facecolor="k"))
    ax.add_patch(Rectangle((70, 30), 1, 40, edgecolor="k", facecolor="k"))

    # Title each graph
    if boundary_func == zero_normal_field_walls:
        ax.set_title("Zero-Normal-Field Walls")
    else:
        ax.set_title("Grounded Conducting Walls")

    ax.get_figure().suptitle(
        "Electrostatic Field Potential\nSurrounding a Parallel Plate Capacitor"
    )
    ax.set_aspect("equal")


def main():
    plt.figure(Path(__file__).name, figsize=(10, 5.5))
    solve_laplace(plt.subplot(1, 2, 1), zero_normal_field_walls)
    solve_laplace(plt.subplot(1, 2, 2), grounded_walls)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
