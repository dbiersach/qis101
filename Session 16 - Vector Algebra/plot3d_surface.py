#!/usr/bin/env -S uv run
"""plot3d_surface.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


def main():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    z = f(x, y)

    plt.figure(Path(__file__).name)
    ax = plt.axes(projection="3d")

    surf = ax.plot_surface(x, y, z, cmap="coolwarm", lw=0, antialiased=False)
    plt.colorbar(surf, ax=ax, shrink=0.5)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter("{x:.02f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    main()
