#!/usr/bin/env -S uv run
"""plot_parabola_instructor.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(ax):
    x = np.linspace(-4, 5)
    y = np.power(x, 2) + 1.0
    ax.plot(x, y, color="olivedrab")
    ax.plot(0, 1, color="red", marker="o")
    ax.axhline(1, color="gray", linestyle="--")
    ax.set_title("$y=x^2+1$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 30)
    ax.grid("on")


def main():
    plt.figure(Path(__file__).name)
    plot(plt.axes())
    plt.show()


if __name__ == "__main__":
    main()
