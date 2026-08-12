#!/usr/bin/env -S uv run
"""plot_quintic.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(ax):
    x = np.linspace(-10, 12, 100)
    y = (x - 11) * (x - 5) * (x + 1) * (x + 4) * (x + 9)
    ax.plot(x, y, c="springgreen", lw=2)
    ax.set_title("$y=x^5-2x^4-120x^3+22x^2+2119x+1980$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid("on")


def main():
    plt.figure(Path(__file__).name, figsize=(9, 6))
    plot(plt.axes())
    plt.show()


if __name__ == "__main__":
    main()
