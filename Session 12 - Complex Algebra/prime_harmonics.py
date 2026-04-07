#!/usr/bin/env -S uv run
"""prime_harmonics.py"""

import lzma
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def H(x: np.ndarray, zeta_zeros: np.ndarray) -> np.ndarray:
    """
    Returns the amplitudes of the frequencies of the Riemann Zeta function,
    which have spikes at all primes and their respective powers
    The explicit formula connecting primes to zeta zeros involves terms
    of the form x^(it), where x is a real number and t is a zeta zero
    """
    # np.outer() pairs every log(x[i]) with every zeta_zeros[j],
    # producing an (N, M) matrix without needing to reshape x
    cos_matrix = np.cos(np.outer(np.log(x), zeta_zeros))

    # Sum across each row (axis=1): add all M cosine contributions for each x[i],
    # producing one output value per x-value with final shape (N,)
    return 1 + np.sum(cos_matrix, axis=1)


def main():
    # Read pickle file containing the zeta zeros
    file_path = Path(__file__).parent / "zeta_zeros.pickle.xz"
    with lzma.open(file_path, "rb") as file_in:
        zeta_zeros = pickle.load(file_in)

    # Sum the waves caused by each Zeta Zero
    x = np.linspace(1.1, 50, 10_000)
    y = H(x, zeta_zeros)

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

    # Plot labelled vertical lines at just the first prime and its first
    # higher power (square) to populate the legend
    prime, powers = next(iter(prime_dict.items()))
    plt.axvline(x=prime, ymax=0.95, linestyle="--", color="red", label="Prime")
    if len(powers) > 1:
        plt.axvline(
            x=powers[1], ymax=0.95, linestyle="--", color="green", label="Prime power"
        )

    # Plot the remaining primes and their powers but without labels
    # so the legend is not overpopulated
    for prime, powers in prime_dict.items():
        plt.axvline(x=prime, ymax=0.95, linestyle="--", color="red")
        for prime_power in powers[1:]:
            plt.axvline(x=prime_power, ymax=0.95, linestyle="--", color="green")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
