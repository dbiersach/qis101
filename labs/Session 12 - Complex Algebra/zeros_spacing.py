#!/usr/bin/env python3
"""zeros_spacing.py"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Read pickle file containing the zeta zeros
file_path = Path(__file__).parent / "zeta_zeros.pickle"
with open(file_path, "rb") as file_in:
    zeta_zeros = pickle.load(file_in)
# Take only the first 1000 zeta zeros
zeta_zeros = zeta_zeros[:1000]

# Create a 1d array of 1000 random floats following a uniform distribution
random_array = sorted(
    np.random.uniform(low=zeta_zeros[0], high=zeta_zeros[-1], size=1000)
)

# Plot the Zeta Zeros in 10 lines of 100 values each
once = True
plt.figure(Path(__file__).name)
for i in range(10):
    start = i * 100
    stop = start + 99
    x = zeta_zeros[start:stop]
    x = x - x[0]
    y = np.full_like(x, i * 10 + 2)
    if once:
        plt.scatter(x[0], y[0], c="b", s=3, label="Zeta Zeros")
        once = False
    plt.scatter(x, y, c="b", s=3)

# Plot the random numbers in 10 lines of 100 values each
once = True
for i in range(10):
    start = i * 100
    stop = start + 99
    x = random_array[start:stop]
    x = x - x[0]
    y = np.full_like(x, i * 10 + 4)
    if once:
        plt.scatter(x[0], y[0], c="r", s=3, label="Uniform Random\nNumbers")
        once = False
    plt.scatter(x, y, c="r", s=3)

plt.title("Zeta Zero Spacing vs. Uniform Random Number Distribution")
plt.xlabel("Imaginary Component of each Zeta Root")
plt.ylabel("100 values per row")
plt.xlim(0, 100)
plt.xticks([])
plt.legend(loc="upper right", framealpha=1.0, facecolor="white")
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.show()
