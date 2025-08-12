#!/usr/bin/env python3
"""fermat_derivative.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def f(x):
    return np.cos(x)


def f_prime_analytic(x):
    return -np.sin(x)


def f_prime_fermat(x, h):
    return (f(x + h) - f(x)) / h


def f_prime_centered(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def main():
    a, b = -4 * np.pi, 4 * np.pi
    n = 500
    h = (b - a) / n
    x = np.linspace(a, b, n)

    y_prime_actual = f_prime_analytic(x)
    y_prime_fermat = f_prime_fermat(x, h)
    y_prime_centered = f_prime_centered(x, h)

    mae_fermat = np.mean(np.abs(y_prime_fermat - y_prime_actual))
    mae_centered = np.mean(np.abs(y_prime_centered - y_prime_actual))

    print(f"Fermat Quotient MAE     : {mae_fermat:>9.7f}")
    print(f"Centered Difference MAE : {mae_centered:>9.7f}")

    plt.figure(Path(__file__).name)
    plt.plot(x, f(x), label=r"$y=\cos{x}$")
    plt.plot(x, y_prime_centered, label=r"$\dfrac{dy}{dx}=-\sin{x}$")
    plt.title("Centered Difference Formula")
    plt.axhline(0, color="black", linestyle="-")
    plt.axvline(0, color="black", linestyle="-")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()
