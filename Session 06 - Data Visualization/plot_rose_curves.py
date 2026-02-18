#!/usr/bin/env -S uv run
"""plot_rose_curves.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def r1(theta):
    return 4 + 4 * np.cos(4 * theta)


def r2(theta):
    return 3 + 3 * np.cos(4 * theta + np.pi)


def r3(theta):
    return 5 + 5 * np.cos(3 / 2 * theta)


def plot(ax):
    theta = np.linspace(0, 4 * np.pi, 1000)
    ax.plot(theta, r1(theta), label=r"$4+4\cos{4\theta}$")
    # ax.plot(theta, r2(theta), label=r"$3+3\cos{(4\theta+\pi)}$")
    # ax.plot(theta, r3(theta), label=r"$5+5\cos{(\dfrac{3}{2}\theta)}$")
    ax.legend(loc="upper right")
    ax.axis("on")


def main():
    plt.figure(Path(__file__).name)
    plot(plt.axes(projection="polar"))
    plt.show()


if __name__ == "__main__":
    main()
