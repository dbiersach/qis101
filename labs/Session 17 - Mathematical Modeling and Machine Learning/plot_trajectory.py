#!/usr/bin/env python3
"""plot_trajectory.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fit_linear(x, y):
    m = len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)
    m = m / (len(x) * np.sum(x**2) - np.sum(x) ** 2)
    b = (np.sum(y) - m * np.sum(x)) / len(x)
    return m, b


def main():
    # Read experiment data from data file
    file_path = Path(__file__).parent / "ray.csv"
    # time in nanoseconds, height in centimeters
    t, h = np.genfromtxt(file_path, delimiter=",", unpack=True)

    # Calculate line of best fit
    slope, yint = fit_linear(t, h)

    # Calculate origination height (oh) and initial velocity (v)
    oh = (slope * 1e9 / 100) * (0.1743 / 1e3) / 1000
    c = 29.98  # speed of light in cm/ns
    v = slope / c

    plt.figure(Path(__file__).name)
    plt.scatter(t, h)
    plt.plot(t, slope * t + yint, color="red", linewidth=2)
    plt.title(
        (
            "Secondary Cosmic Ray Trajectory\n"  # noqa
            rf"$m={slope:.2f}\,\frac{{cm}}{{ns}}\quad$"
            rf"$v_0={v:.2f}\,\frac{{m}}{{s}}\quad$"
            rf"$oh={oh:,.2f}\,km$"
        )
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Detector Height (cm)")
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
