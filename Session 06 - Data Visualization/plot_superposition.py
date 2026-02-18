#!/usr/bin/env -S uv run
"""plot_superposition.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(ax):
    theta = np.linspace(0, 4 * np.pi, 1000)
    radius = 7 + 7 * np.sin(11 * theta) * np.cos(5 * theta)
    ax.plot(
        theta,
        radius,
        color="black",
        label=r"$7+7\,\sin{(11\,\theta)}\,\cos{(5\,\theta)}$",
    )
    ax.legend(loc="upper right")
    ax.axis("on")


def main():
    plt.figure(Path(__file__).name)
    plot(plt.axes(projection="polar"))
    plt.show()


if __name__ == "__main__":
    main()
