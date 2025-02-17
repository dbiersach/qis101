#!/usr/bin/env python3
"""fermat_derivative.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def F(x):
    return np.cos(x)


def f_analytic(x):
    return -np.sin(x)


def f_fermat_quotient(x, h):
    return (F(x + h) - F(x)) / h


def f_central_difference(x, h):
    return (F(x + h) - F(x - h)) / (2 * h)


def main():
    a, b = -4 * np.pi, 4 * np.pi
    n = 500
    h = (b - a) / n
    x = np.linspace(a, b, n)

    y_actual = f_analytic(x)
    y_fermat = f_fermat_quotient(x, h)
    y_central = f_central_difference(x, h)

    print(f"Fermat Quotient Error    : {sum((y_actual - y_fermat) ** 2):>9.7f}")
    print(f"Central Difference Error : {sum((y_actual - y_central) ** 2):>9.7f}")

    plt.figure(Path(__file__).name)
    plt.plot(x, F(x), label=r"$y=\cos{x}$")
    plt.plot(x, y_central, label=r"$\dfrac{dy}{dx}=-\sin{x}$")
    plt.title("Central Difference Formula")
    plt.axhline(0, color="black", linestyle="-")
    plt.axvline(0, color="black", linestyle="-")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()
