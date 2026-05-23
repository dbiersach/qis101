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


def main() -> None:
    file_path = Path(__file__).parent / "scintillator_pulse.csv"
    # The CSV stores time in nanoseconds and signal amplitude in volts
    time_ns, volts = np.genfromtxt(file_path, delimiter=",", unpack=True)
    # Use microseconds for plotting because it is easier to read on the x-axis
    time_us = time_ns / 1000.0
    # Use seconds for physical unit conversion during integration
    time_s = time_ns * 1e-9
    # Convert voltage to current using the detector/effective resistance
    # I = V / R, so integrating current over time gives charge in coulombs
    resistance_ohms = 1000.0
    current_a = volts / resistance_ohms
    # Calculate cumulative collected charge
    charge_c = cumulative_simpson(current_a, x=time_s, initial=0)
    # Plot charge in nanocoulombs for readable axis values
    charge_nc = charge_c * 1e9
    # Calculate the region of interest from the cumulative charge curve
    roi = widest_span_with_greatest_slope(time_us, charge_nc)

    plt.figure(Path(__file__).name, figsize=(12, 5))

    ax = plt.subplot(1, 2, 1)
    ax.scatter(time_us, volts, s=0.5)
    ax.set_title("Photon Scintillation")
    ax.set_xlabel(r"Time $(\mu s)$")
    ax.set_ylabel("Voltage (V)")

    ax = plt.subplot(1, 2, 2)
    ax.plot(time_us, charge_nc)
    ax.plot(time_us[roi], charge_nc[roi], c="r", lw=2)
    ax.set_title("Collected Charge (Region of Interest)")
    ax.set_xlabel(r"Time $(\mu s)$")
    ax.set_ylabel("Charge (nC)")

    plt.show()


if __name__ == "__main__":
    main()
