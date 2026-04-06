#!/usr/bin/env -S uv run
"""scintillator_pulse.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson


def widest_span_with_greatest_slope(x, y, cutoff=0.5):
    # Compute slopes between adjacent points only
    slopes = np.abs(np.diff(y) / np.diff(x))
    threshold = np.max(slopes) * cutoff

    # Find widest span where adjacent slope exceeds threshold
    indices = np.where(slopes > threshold)[0]
    min_i, max_j = indices[0], indices[-1] + 1

    # Use boolean indexing to create a mask for the region of interest
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
