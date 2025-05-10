#!/usr/bin/env python3
"""zeta_harmonics.py"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath
import numpy as np
from matplotlib.ticker import MultipleLocator

prime_dict = {}


def F(x):
    """
    Returns the real cosine frequencies of the Riemann Zeta function,
    which are precisely the imaginary components of the complex Zeta zeros
    """
    s: float = 0
    for prime in prime_dict:
        powers: list[int] = prime_dict[prime]
        for power in powers:
            s += -(np.log(prime) / np.sqrt(power) * np.cos(x * np.log(power)))
    return s


def main():
    # Read pickle file containing the prime powers dictionary
    file_path = Path(__file__).parent / "prime_powers.pickle"
    with open(file_path, "rb") as file_in:
        global prime_dict
        prime_dict = pickle.load(file_in)

    x = np.linspace(0, 100, 1_000)
    vectorized_F = np.vectorize(F, excluded=["prime_dict"])
    y = vectorized_F(x)

    plt.figure(Path(__file__).name)
    plt.plot(x, y, zorder=3)
    plt.title("Zeta Harmonics from the Prime Powers")
    plt.xlabel("Imaginary Components of the Complex Zeros of Riemann's Zeta function")
    plt.ylabel("Amplitude")
    plt.xlim(10, 100)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # fmt: off
    plt.axvline(x=mpmath.zetazero(1).imag, ymax=0.9,
        linestyle="--", color="red", label="Zeta Zero")
    # fmt: on

    for i in range(2, 30):
        plt.axvline(x=mpmath.zetazero(i).imag, ymax=0.9, linestyle="--", color="red")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
