#!/usr/bin/env -S uv run
"""euler_formula.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_exponential_curve(ax):
    # Plot y = e^x
    x = np.linspace(-10, 10, 1000)
    ax.plot(x, np.exp(x), color="green", label=r"$e^x$")


def plot_euler_formula(ax):
    # Plot y = e^xi
    x = np.linspace(-10, 10, 1000)
    z = np.zeros(len(x), dtype=complex)

    for idx, val in enumerate(x):
        z[idx] = np.exp(complex(0, val))

    ax.plot(np.real(z), np.imag(z), color="blue", linewidth=2, label=r"$e^{i x}$")


def plot_complex_point1(ax):
    # Plot a complex exponential on this Argand diagram
    z = 1.7 * np.exp(complex(0, 0.62))
    x, y = np.real(z), np.imag(z)
    ax.scatter(x, y, color="black")

    line_hypot = [(0, 0), (x, y)]
    line_opp = [(x, 0), (x, y)]
    line_adj = [(0, 0), (x, 0)]
    lc = LineCollection([line_hypot, line_opp, line_adj], color="red", zorder=3)
    ax.add_collection(lc)

    # fmt: off
    ax.annotate(r"$1.7e^{0.62 i}$", xy=(x, y), 
                color="black", xytext=(-15, 10), textcoords="offset pixels")

    ax.annotate(r"$1.7\sin{0.62}$", xy=(x, y / 2),
                color="red", xytext=(5, 0), textcoords="offset pixels")

    ax.annotate(r"$1.7\cos{0.62}$", xy=(x / 3, 0),
                color="red", xytext=(0, 5), textcoords="offset pixels")
    # fmt: on


def plot_complex_point2(ax):
    # Plot -1-1.5j as complex exponential
    x, y = -1.0, -1.5
    z = complex(x, y)
    hypot = np.hypot(np.real(z), np.imag(z))
    theta = np.arctan(np.imag(z) / np.real(z)) - np.pi
    ax.scatter(-1, -1.5, color="purple")
    line_hypot = [(0, 0), (x, y)]
    line_opp = [(x, 0), (x, y)]
    line_adj = [(0, 0), (x, 0)]
    lc = LineCollection([line_hypot, line_opp, line_adj], color="purple", zorder=3)
    ax.add_collection(lc)
    # fmt: off
    ax.annotate(
        rf"$({x}{y}i)={hypot:.3f}e^{{{theta:.3f}i}}$", xy=(x, y),
        color="purple", xytext=(-55, -25), textcoords="offset pixels")
    # fmt: on


def main():
    plt.figure(Path(__file__).name)
    ax = plt.gca()

    plot_exponential_curve(ax)
    plot_euler_formula(ax)
    plot_complex_point1(ax)
    plot_complex_point2(ax)

    plt.title(r"Euler's Formula: $z = e^{\pm i\theta}=\cos{\theta}\pm i\sin{\theta}$")
    plt.xlabel("Real z")
    plt.ylabel("Imaginary z")
    plt.axvline(x=0, color="black", linewidth=2)
    plt.axhline(y=0, color="black", linewidth=2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    ax.legend(loc="upper right")
    ax.grid("on")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
