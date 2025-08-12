#!/usr/bin/env python3
"""task01_01.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def f1(n):
    """Dave Biersach's Solution"""
    n = np.asarray(n, dtype=np.complex128)  # Ensure complex type
    s = np.power(np.sin(n * np.pi / 2), 2)
    denominator = n + s

    # Avoid division by zero using np.where and complex nan
    result = np.where(
        denominator != 0,
        2 * np.power(-1 + 0j, n + 1) * np.power(2, s) / denominator,
        np.nan + 0j,  # Return complex NaN where denominator is 0
    )

    return result


def f2(n):
    """Cameron Senna's Solution"""
    n = np.asarray(n, dtype=np.complex128)  # Ensure complex type
    s = np.cos(np.pi * n)
    denominator = n + 1
    denominator2 = n

    # Avoid division by zero using np.where and complex nan
    result = np.where(
        (denominator != 0) | (denominator2 != 0),
        2 / denominator * (1 - s) - (1 / denominator2) * (1 + s),
        np.nan + 0j,  # Return complex NaN where denominator is 0
    )

    return result


def main():
    plt.figure(Path(__file__).name)
    xd = np.arange(0, 13)
    plt.scatter(xd, f1(xd), color="black", label="D. Biersach")
    plt.scatter(xd, f2(xd), color="red", label="C. Senna")
    xc = np.linspace(0, 13, 500)
    y1 = f1(xc)
    y2 = f2(xc)
    plt.plot(xc, np.real(y1), label="D.Biersach (real)")
    plt.plot(xc, np.real(y2), label="C. Senna (real)")
    plt.plot(xc, np.imag(y1), label="D. Biersach (imag)")
    plt.plot(xc, np.imag(y2), label="C.Senna (imag)")
    plt.xlim(0, 13)
    plt.ylim(-1.5, 2.5)
    plt.title("Task 01-01 Generators")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid("on")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_axisbelow(True)
    plt.show()


if __name__ == "__main__":
    main()
