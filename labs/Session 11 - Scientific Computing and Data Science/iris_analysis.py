#!/usr/bin/env python3
"""iris_analysis.py"""

# Iris CSV taken from https://archive.ics.uci.edu/dataset/53/iris

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
file_name = "iris.csv"
file_path = Path(__file__).parent / file_name
petal_data = np.genfromtxt(
    file_path, delimiter=",", skip_header=1, usecols=(2, 3), dtype=float
)
species_data = np.genfromtxt(
    file_path, delimiter=",", skip_header=1, usecols=(4), dtype=str
)

# Plot the relationship between petal length and petal width between varieties
plt.figure(Path(__file__).name, figsize=(8, 6))
for species in np.unique(species_data):
    mask = species_data == species  # Broadcasting and element-wise comparison
    plt.scatter(petal_data[mask, 0], petal_data[mask, 1], label=species.strip('"'))
plt.title("Iris Petal Length vs Petal Width By Species")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend(loc="upper left")
plt.show()
