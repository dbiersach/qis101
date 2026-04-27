#!/usr/bin/env -S uv run
"""bernoulli_series.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray


def integrand_substituted(u: NDArray) -> NDArray:
    """Return 2u * u^(-2u^2), the integrand of x^(-x) from 0 to 1 after x = u^2.

    The substitution x = u^2 maps the vertical tangent of x^(-x) at x = 0
    into a smooth linear zero at u = 0, restoring effectively exponential
    convergence for Gauss-Legendre quadrature.
    """
    return 2.0 * u * u ** (-2.0 * u * u)


def gauss_legendre_substituted(n: int) -> float:
    """Approximate the integral of x^(-x) from 0 to 1 with n-point Gauss-Legendre after x = u^2."""

    # n-point Gauss-Legendre rule on the reference interval [-1, 1]
    nodes, weights = leggauss(n)
    # affine map from [-1, 1] to [0, 1]
    u = 0.5 * (nodes + 1.0)
    # leading 0.5 is the Jacobian (b - a) / 2
    return float(0.5 * np.sum(weights * integrand_substituted(u)))


def main() -> None:
    terms = 14
    x = np.arange(1, terms + 1)
    y1 = np.zeros_like(x, dtype=np.float64)  # series partial sums
    y2 = np.zeros_like(x, dtype=np.float64)  # quadrature estimates

    # See https://en.wikipedia.org/wiki/Sophomore%27s_dream
    for n in range(1, terms):
        y1[n] = y1[n - 1] + n ** (-n)
        y2[n] = gauss_legendre_substituted(n)

    plt.figure(Path(__file__).name)
    plt.plot(x, y1, label=r"$\sum_{n=1}^{N} n^{-n}$")
    plt.plot(x, y2, label=r"$\int_0^1 x^{-x}\,dx$")
    plt.title("Johann Bernoulli's Identity (1697)")
    plt.xlabel("Number of Terms / Quadrature Nodes")
    plt.ylabel("Approximate Sum")
    plt.annotate(
        f"{y1[-1]:.14f}",
        (x[-1], y1[-1]),
        xytext=(-5, -15),
        textcoords="offset points",
        ha="right",
        color="C0",
    )
    plt.annotate(
        f"{y2[-1]:.14f}",
        xy=(x[-1], y2[-1]),
        xytext=(-5, 10),
        textcoords="offset points",
        ha="right",
        color="C1",
    )
    plt.legend(loc="center right")
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
