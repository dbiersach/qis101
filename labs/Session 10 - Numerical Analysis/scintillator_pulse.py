#!/usr/bin/env python3
"""scintillator_pulse.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.integrate import cumulative_simpson


@njit
def widest_span_with_greatest_slope(x, y, cutoff=0.5):
    # Find the maximum slope between any adjacent points
    max_slope = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x) - 1):
            slope = (y[j] - y[i]) / (x[j] - x[i])
            if abs(slope) > max_slope:
                max_slope = abs(slope)
    min_i = len(x)
    max_j = 0
    threshold = max_slope * cutoff
    # Find widest span having slope within the cutoff
    for i in range(len(x)):
        for j in range(i + 1, len(x) - 1):
            slope = (y[j] - y[i]) / (x[j] - x[i])
            if abs(slope) > threshold:
                if i <= min_i:
                    min_i = i
                if j >= max_j:
                    max_j = j
    # Return array of indexes within region of interest
    # using Numpy's Boolean Indexing
    roi = (x > x[min_i]) & (x < x[max_j])
    return roi


def main():
    file_path = Path(__file__).parent / "scintillator_pulse.csv"
    time, volts = np.genfromtxt(file_path, delimiter=",", unpack=True)
    time /= 1000  # Scale domain to microseconds

    # Calculate the region of interest
    charge = cumulative_simpson(volts, x=time, initial=0)
    roi = widest_span_with_greatest_slope(time, charge)

    plt.figure(Path(__file__).name, figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    ax.scatter(time, volts, s=0.5)
    ax.set_title("Photon Scintillation")
    ax.set_xlabel(r"Time $(\mu s)$")
    ax.set_ylabel("Voltage (V)")

    ax = plt.subplot(1, 2, 2)
    ax.plot(time, charge)
    ax.plot(time[roi], charge[roi], c="r", lw=2)
    ax.set_title("Deposited Energy (Region of Interest)")
    ax.set_xlabel(r"Time $(\mu s)$")
    ax.set_ylabel("Coulomb (C)")

    plt.show()


if __name__ == "__main__":
    main()
