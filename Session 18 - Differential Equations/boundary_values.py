#!/usr/bin/env -S uv run
"""boundary_values.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp


def model(x, state_vector):
    # Unpack current state vector (dependent variables)
    y, dy = state_vector
    # Express y'' + 4y = 0 as a first-order system
    d_y = dy
    d_dy = -4 * y
    return np.vstack((d_y, d_dy))


def boundary_conditions(ya, yb):
    # ya = state vector at x = x_initial, yb = state vector at x = x_final
    # Residuals must vanish when y(0) = -2 and y(pi/4) = -3
    return np.array([ya[0] + 2, yb[0] + 3])


def main():
    # Define domain endpoints
    x_initial = 0.0
    x_final = np.pi / 4

    # Build coarse initial mesh and a flat initial guess for [y, y']
    # solve_bvp will refine the mesh adaptively
    x_mesh = np.linspace(x_initial, x_final, 11)
    y_guess = np.zeros((2, x_mesh.size))

    # Numerically solve the boundary value problem
    sol = solve_bvp(model, boundary_conditions, x_mesh, y_guess)
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    # Sample the numerical solution on a fine grid for plotting
    x_plot = np.linspace(x_initial, x_final, 200)
    y_numeric = sol.sol(x_plot)[0]

    # Analytic solution y = -2 cos(2x) - 3 sin(2x) for comparison
    y_exact = -2 * np.cos(2 * x_plot) - 3 * np.sin(2 * x_plot)

    plt.figure(Path(__file__).name)
    plt.plot(x_plot, y_exact, lw=6, color="lightgray", label="Analytic")
    plt.plot(x_plot, y_numeric, lw=2, color="crimson", label="solve_bvp")
    plt.scatter(
        [x_initial, x_final],
        [-2, -3],
        color="black",
        zorder=5,
        label="Boundary conditions",
    )
    plt.title(r"BVP: $y'' + 4y = 0,\ \ y(0) = -2,\ \ y(\pi/4) = -3$")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend(framealpha=1.0, facecolor="white")
    plt.grid("On")
    plt.show()


if __name__ == "__main__":
    main()
