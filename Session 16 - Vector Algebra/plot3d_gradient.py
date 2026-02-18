#!/usr/bin/env -S uv run
"""plot3d_gradient.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


def main():
    coords = np.linspace(-5, 5, 100)
    xf, yf = np.meshgrid(coords, coords)  # fine grid
    xc, yc = np.meshgrid(coords[::5], coords[::5])  # coarse grid
    zf, zc = f(xf, yf), f(xc, yc)
    dy, dx = np.gradient(zc)  # gradient along coarse grid

    plt.figure(Path(__file__).name, figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    ax.contourf(xf, yf, zf, 100, cmap="coolwarm")  # fine grid
    # The gradient points in the direction of greatest ASCENT
    ax.quiver(xc, yc, dx, dy, color="k")  # coarse grid
    ax.axis("off")
    ax = plt.subplot(1, 2, 2, projection="3d")
    ax.plot_surface(xf, yf, zf, cmap="coolwarm", lw=0, antialiased=False)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
