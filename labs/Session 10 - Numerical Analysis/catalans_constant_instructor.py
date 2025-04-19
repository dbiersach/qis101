#!/usr/bin/env python3
"""catalans_constant_instructor.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate


def f(x):
    return np.log(np.sin(x) + np.cos(x))


def catalans_constant():
    """Returns Catalan's Constant G"""
    n = np.arange(1_000_000)
    x = (-1) ** n
    y = (2 * n + 1) ** 2
    return np.sum(x / y)


def main():
    # From https://en.wikipedia.org/wiki/Catalan%27s_constant

    G = catalans_constant()  # Naive slow converging method
    print(f"{G=:.15f} - Correct to 3 digits")

    G = 0.915965594177219  # Using this would be cheating!
    print(f"{G=:.15f} - Correct to 15 digits")

    # Use one of the faster converging integral identities
    G = scipy.integrate.quad(lambda x: np.log(1 / np.tan(x)), 0, np.pi / 4)[0]
    print(f"{G=:.15f} - Correct to 15 digits")

    # Numerically estimate the given integral
    est_area = scipy.integrate.quad(f, 0, np.pi / 2)[0]
    print(f"{est_area=:.15f}")

    # Calculate the requested difference
    diff_val = G - (np.pi * np.log(2) / 4)
    print(f"{diff_val=:.15f}")

    # Verified using Wolfram Mathematica 14.2
    act_area = 0.371569071601318
    print(f"{act_area=:.15f}")

    plt.figure(Path(__file__).name)
    x = np.linspace(-0.17, 1.74, 1000)
    y = f(x)
    plt.plot(x, y, lw=2)
    plt.fill_between(x, y, where=(y >= 0), color="r", alpha=0.3)
    plt.title(r"$y=\ln(\sin x+\cos x)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
