#!/usr/bin/env python3
"""prime_harmonics.py"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

zeta_zeros = {}


def f(s: float) -> float:
    """
    Returns the amplitudes of the frequencies of the Riemann Zeta function,
    which have spikes at all primes and their respective powers
    """
    return 1 + np.sum(np.cos(np.log(s) * zeta_zeros))


def main():
    global zeta_zeros

    # Read pickle file containing the zeta zeros
    file_path = Path(__file__).parent / "zeta_zeros.pickle"
    with open(file_path, "rb") as file_in:
        zeta_zeros = pickle.load(file_in)

    # Read pickle file containing the prime powers dictionary
    file_path = Path(__file__).parent / "prime_powers.pickle"
    with open(file_path, "rb") as file_in:
        prime_dict = pickle.load(file_in)
        vectorized_H = np.vectorize(f)
        x = np.linspace(1.1, 50, 10_000)
        y = vectorized_H(x)

    plt.figure(Path(__file__).name)
    plt.plot(x, y, zorder=3)
    plt.title("Prime Powers from the Zeta Harmonics")
    plt.xlabel("Spectrum of the Complex Zeros of Riemann's Zeta function")
    plt.ylabel("Amplitude")
    plt.xlim(x[0], x[-1])
    plt.ylim(0, max(y) * 1.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))

    for prime in prime_dict:
        plt.axvline(x=prime, ymax=0.95, linestyle="--", color="red")
        powers = prime_dict[prime][1:]
        for prime_power in powers:
            plt.axvline(x=prime_power, ymax=0.95, linestyle="--", color="green")

    plt.show()


if __name__ == "__main__":
    main()
