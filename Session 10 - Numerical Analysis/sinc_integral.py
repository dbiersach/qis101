#!/usr/bin/env -S uv run
"""sinc_gauss_legendre.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray


def sinc(x: NDArray) -> NDArray:
    """Return sin(x) / x, with the removable value sinc(0) = 1."""
    return np.where(x == 0.0, 1.0, np.sin(x) / x)


def gauss_legendre_sinc(n: int, a: float, b: float) -> float:
    """Approximate the integral of sinc(x) from a to b
    with n-point Gauss-Legendre quadrature."""

    # n-point Gauss-Legendre rule on the reference interval [-1, 1]
    nodes, weights = leggauss(n)
    # affine map from [-1, 1] to [a, b]
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    # leading factor is the Jacobian (b - a) / 2
    return float(0.5 * (b - a) * np.sum(weights * sinc(x)))


def main() -> None:
    nodes = 1_000
    a, b = 0.0, 10.0 * np.pi

    x = np.linspace(a, b, nodes)
    y = sinc(x)
    area = gauss_legendre_sinc(nodes, a, b * 50)

    plt.figure(Path(__file__).name)
    plt.plot(x, y, label=r"$\dfrac{\sin(x)}{x}$")
    plt.axhline(0.0, linewidth=1.0)
    plt.fill_between(x, y, 0.0, alpha=0.20)
    plt.title(r"Sinc Function and Gauss-Legendre Area")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\dfrac{\sin(x)}{x}$")
    x, y = 0.98 * b, sinc(np.array([0.98 * b]))[0]
    # fmt:off
    plt.annotate(f"area ≈ {area:.14f}", xy=(x, y),
        xytext=(-10, 22), textcoords="offset points",
        ha="right", color="C0")
    plt.annotate(f"2 * area ≈ {2 * area:.14f}", xy=(x, y),
        xytext=(-10, 35), textcoords="offset points",
        ha="right", color="C1")
    # fmt:on
    plt.legend(loc="upper right")
    plt.gca().xaxis.set_major_locator(MultipleLocator(np.pi))
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
