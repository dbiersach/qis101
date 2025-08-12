#!/usr/bin/env python3
"""plot_parabola.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(ax):
    x = np.linspace(-4, 5)
    y = np.power(x, 2) + 1.0
    ax.plot(x, y, color="olivedrab")


def main():
    plt.figure(Path(__file__).name)
    plot(plt.axes())
    plt.show()


if __name__ == "__main__":
    main()
