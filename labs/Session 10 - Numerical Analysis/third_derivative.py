#!/usr/bin/env python3
"""third_derivative.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def f(x):
    return np.array(np.sin(x**2) / (1 + x**3))


def main():
    a, b, n = 0, 5, 500
    dx = (b - a) / n

    x = np.linspace(a, b, n)
    y = f(x)

    y_prime = np.zeros_like(y)
    for i in range(1, len(y) - 1):
        y_prime[i] = (y[i + 1] - y[i - 1]) / (2 * dx)

    y_prime2 = np.zeros_like(y)
    for i in range(2, len(y_prime2) - 2):
        y_prime2[i] = (y_prime[i + 1] - y_prime[i - 1]) / (2 * dx)

    y_prime3 = np.zeros_like(y)
    for i in range(3, len(y_prime3) - 3):
        y_prime3[i] = (y_prime2[i + 1] - y_prime2[i - 1]) / (2 * dx)

    plt.figure(Path(__file__).name)
    plt.plot(x, 20 * y, label=r"$20\times\frac{\sin{x^2}}{1+x^3}$")
    plt.plot(
        x[1:-2],
        10 * y_prime[1:-2],
        label=r"$10\times\frac{\partial}{\partial x}\;\frac{\sin{x^2}}{1+x^3}$",
    )
    plt.plot(
        x[2:-3],
        y_prime2[2:-3],
        label=r"$\frac{\partial ^2}{\partial x^2}\;\frac{\sin{x^2}}{1+x^3}$",
    )
    plt.plot(
        x[3:-4],
        y_prime3[3:-4],
        label=r"$\frac{\partial ^3}{\partial x^3}\;\frac{\sin{x^2}}{1+x^3}$",
    )
    plt.title(r"Higher Order Numerical Derivatives of $\frac{\sin{x^2}}{1+x^3}$")
    plt.xlim(x[0], x[-1])
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linestyle="-")
    plt.axvline(0, color="black", linestyle="-")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
