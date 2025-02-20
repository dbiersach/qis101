# plot_trajectory.py

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
    file_name = "ray.csv"
    file_path = Path(__file__).parent / file_name
    data = np.genfromtxt(file_path, delimiter=",")
    # time in nanoseconds, height in centimeters
    t, h = data.T

    # Calculate line of best fit
    slope, yint = fit_linear(t, h)

    # Calculate origination height (oh) and initial velocity (v)
    oh = (slope * 1e9 / 100) * (0.1743 / 1e3) / 1000
    c = 29.98  # speed of light in cm/ns
    v = slope / c

    print(f"Slope = {slope:.4f} cm/ns")
    print(f"Velocity = {v:.2f} c")
    print(f"Origination Height = {oh:,.2f} km")

    plt.figure(Path(__file__).name)
    plt.scatter(t, h)
    plt.plot(t, slope * t + yint, color="red", linewidth=2)
    plt.title(
        "Secondary Cosmic Ray Trajectory\n"
        f"Velocity = {v:.2f}c "
        f"Origination Height = {oh:,.2f}km"
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Detector Height (cm)")
    plt.grid("on")
    plt.show()


main()
