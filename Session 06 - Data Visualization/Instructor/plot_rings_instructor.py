#!/usr/bin/env -S uv run
"""plot_rings_instructor.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(ax):
    # Draw the Olympic Rings
    radius = 25.0
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    x_offset = 5 / 2 * radius
    y_offset = radius

    ax.plot(x, y, color="black", linewidth=12)
    ax.plot(x - x_offset, y, color="blue", linewidth=12)
    ax.plot(x + x_offset, y, color="red", linewidth=12)
    ax.plot(x - x_offset / 2, y - y_offset, color="yellow", linewidth=12)
    ax.plot(x + x_offset / 2, y - y_offset, color="green", linewidth=12)

    ax.set_title("The Olympic Rings")
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    plt.figure(Path(__file__).name)
    plot(plt.axes())
    plt.show()


if __name__ == "__main__":
    main()
