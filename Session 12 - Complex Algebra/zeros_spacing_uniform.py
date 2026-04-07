#!/usr/bin/env -S uv run
"""zeros_spacing_uniform.py"""

import lzma
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def generate_random_numbers(n, low, high):
    # Create a 1d array of n random floats following a uniform distribution
    # within the span of the smallest and largest Zeta Zeros
    return np.sort(np.random.uniform(low=low, high=high, size=n))


def main():
    # Read pickle file containing the zeta zeros
    file_path = Path(__file__).parent / "zeta_zeros.pickle.xz"
    with lzma.open(file_path, "rb") as file_in:
        zeta_zeros = pickle.load(file_in)
        # Take only the first 1000 zeta zeros
        zeta_zeros = zeta_zeros[:1000]

    # Generate 1000 random numbers spanning the range of the zeta zeros
    random_array = generate_random_numbers(1000, low=zeta_zeros[0], high=zeta_zeros[-1])

    # Plot the Zeta Zeros in 10 lines of 100 values each
    plt.figure(Path(__file__).name)
    for i in range(10):
        x = zeta_zeros[i * 100 : i * 100 + 100]
        x = x - x[0]
        y = np.full_like(x, i * 10 + 2)
        label = "Zeta Zeros" if i == 0 else None
        plt.scatter(x, y, c="b", s=3, label=label)

    # Plot the random numbers in 10 lines of 100 values each
    for i in range(10):
        x = random_array[i * 100 : i * 100 + 100]
        x = x - x[0]
        y = np.full_like(x, i * 10 + 4)
        label = "Uniform Random\nNumbers" if i == 0 else None
        plt.scatter(x, y, c="r", s=3, label=label)

    plt.title("Zeta Zero Spacing vs. Uniform Random Number Distribution")
    plt.xlabel("Imaginary Component of each Zeta Root")
    plt.ylabel("100 spacings per row")
    plt.xlim(0, 100)
    plt.xticks([])
    plt.legend(loc="upper right", framealpha=1.0, facecolor="white")
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.show()


if __name__ == "__main__":
    main()
