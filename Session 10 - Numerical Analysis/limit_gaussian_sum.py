#!/usr/bin/env -S uv run
"""limit_gaussian_sum.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def f(n: int) -> float:
    # Create an array of integers from 0 to n
    x = np.arange(n + 1)
    # Calculate the sum of the integers from 1 to n and divide by n^2
    return np.sum(x) / n**2


def main() -> None:
    # Create an array of integers from 1 to 999
    x = np.arange(1, 1000)
    # Vectorize the function f so it can be applied to an array
    y = np.vectorize(f)(x)

    # Plot the function
    plt.figure(Path(__file__).name)
    plt.plot(x, y)
    # Plot horizontal asymptote at last y-value
    plt.axhline(y[-1], c="r", ls="--", lw=2, alpha=0.5)
    # Set title and labels
    plt.title(
        r"$f\,\left(n\right)=\dfrac{1}{n^2}\sum_{x=1}^{n}x"
        r"\quad\longrightarrow\quad\dfrac{1}{2}\text{ as } n\to\infty$",
        fontsize=14,
    )
    plt.xlabel(r"$n$", fontsize=12)
    plt.ylabel(r"$f\,\left(n\right)$", fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()
