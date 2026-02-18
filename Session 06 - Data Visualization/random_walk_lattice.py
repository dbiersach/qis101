#!/usr/bin/env -S uv run
"""random_walk_lattice.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def get_avg_dist(dims, max_steps, num_walks):
    """
    Returns the mean final distance (normalized)
    of (num_walks) uniform random walks having length (num_steps)
    on a unit lattice having (dims) dimensions
    """
    avg_dist = np.zeros(max_steps)
    for step in range(max_steps):
        total_dist = 0.0
        for _ in range(num_walks):
            steps = np.zeros(dims)
            for _ in range(step):
                # Take one step in some random dimension
                h = np.random.randint(0, dims)
                steps[h] += -1 if np.random.rand() < 0.5 else 1
            # Calculate straight-line distance traveled
            total_dist += np.sqrt(np.sum(np.power(steps, 2)))
        # Calculate average final distance for this number of steps
        avg_dist[step] = total_dist / num_walks
    return avg_dist


def fit_linear(x, y):
    """
    Returns the slope (m) and y-intercept (b) of the line
    that passes through (x, y) data values with least error
    """
    n = len(x)
    m = float(n * np.sum(x * y) - np.sum(x) * np.sum(y))
    m /= float(n * np.sum(x**2) - np.sum(x) ** 2)
    b = float(np.sum(y) - m * np.sum(x))
    b /= n
    return m, b


def main():
    # Number of dimensions
    dims = 2

    # Walks increase in length from 1 to max_steps
    max_steps = 200

    # Number of times a walk of each length is repeated to find its average
    num_walks = 50_000

    print("This may take up to 30 seconds . . .")
    steps = np.arange(max_steps)
    distances = get_avg_dist(dims, max_steps, num_walks)
    distances_squared = distances**2

    m, b = fit_linear(steps, distances_squared)
    print(f"Slope of line = {m:.4f}")

    plt.figure(Path(__file__).name, figsize=(12, 5))

    ax = plt.subplot(1, 2, 1)
    ax.plot(steps, distances)
    ax.set_title(f"Uniform Random Walk on {dims}-D Unit Lattice")
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Mean Final Distance")

    ax = plt.subplot(1, 2, 2)
    ax.plot(steps, distances_squared, color="green", zorder=3)
    ax.plot(steps, m * steps + b, color="red", linewidth=3)
    ax.set_title(rf"$Slope\;of\;Line\times{{4}}={4 * m:.4f}$")
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel(r"$(Mean\;Final\;Distance)^2$")

    plt.show()


if __name__ == "__main__":
    main()
