#!/usr/bin/env -S uv run
"""plot_circle.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(ax):
    radius = 250
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y)
    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    ax.set_title(f"$x^2 + y^2 = {radius}$")
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_aspect("equal")
    ax.grid("on")


def main():
    plt.figure(Path(__file__).name)
    plot(plt.axes())
    plt.show()


if __name__ == "__main__":
    main()
