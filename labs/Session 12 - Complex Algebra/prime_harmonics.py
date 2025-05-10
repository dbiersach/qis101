#!/usr/bin/env python3
"""prime_harmonics.py"""

import lzma
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

zeta_zeros = {}


def H(x: float) -> float:
    """
    Returns the amplitudes of the frequencies of the Riemann Zeta function,
    which have spikes at all primes and their respective powers
    """
    return 1 + np.sum(np.cos(np.log(x) * zeta_zeros))


def main():
    # Read pickle file containing the zeta zeros
    file_path = Path(__file__).parent / "zeta_zeros.pickle.xz"
    with lzma.open(file_path, "rb") as file_in:
        global zeta_zeros
        zeta_zeros = pickle.load(file_in)

    # Sum the waves caused by each Zeta Zero
    x = np.linspace(1.1, 50, 10_000)
    f = np.vectorize(H)
    y = f(x)

    # Read pickle file containing the prime powers dictionary
    file_path = Path(__file__).parent / "prime_powers.pickle"
    with open(file_path, "rb") as file_in:
        prime_dict = pickle.load(file_in)

    # Plot the spectrum of the non-trivial Zeta Zeros
    plt.figure(Path(__file__).name)
    plt.plot(x, y, zorder=3)
    plt.title("Prime Powers from the Zeta Harmonics")
    plt.xlabel("Spectrum of the Complex Zeros of Riemann's Zeta function")
    plt.ylabel("Amplitude")
    plt.xlim(x[0], x[-1])
    plt.ylim(0, max(y) * 1.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Plot "labelled" vertical lines at just the first prime
    # and its first power to populate the legend
    prime, powers = next(iter(prime_dict.items()))
    power = powers[1]
    # fmt: off
    plt.axvline(x=prime, ymax=0.95,
        linestyle="--", color="red", label="Prime")
    plt.axvline(x=power, ymax=0.95,
        linestyle="--", color="green", label="Prime power")
    # fmt: on

    # Plot the remaining primes and their powers but without labels
    # so the legend is not overpopulated
    for prime in prime_dict:
        plt.axvline(x=prime, ymax=0.95, linestyle="--", color="red")
        powers = prime_dict[prime][1:]
        for prime_power in powers:
            plt.axvline(x=prime_power, ymax=0.95, linestyle="--", color="green")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
