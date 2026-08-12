#!/usr/bin/env -S uv run
"""zeta_harmonics.py"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath
import numpy as np
from matplotlib.ticker import MultipleLocator


def F(x: np.ndarray, amplitudes: np.ndarray, log_powers: np.ndarray) -> np.ndarray:
    """
    Returns the real cosine frequencies of the Riemann Zeta function,
    which are precisely the imaginary components of the complex Zeta zeros.
    Each prime power contributes a cosine wave of frequency log(power)
    and amplitude log(prime)/sqrt(power).
    """
    # np.outer() pairs every x[i] with every log_powers[j],
    # producing an (N, M) matrix of cosine frequency arguments
    cos_matrix = np.cos(np.outer(x, log_powers))

    # Scale each column j by its amplitude, then sum across each row (axis=1):
    # add all M prime power contributions for each x[i],
    # producing one output value per x-value with final shape (N,)
    return -np.sum(amplitudes * cos_matrix, axis=1)


def main():
    # Read pickle file containing the prime powers dictionary
    file_path = Path(__file__).parent / "prime_powers.pickle"
    with open(file_path, "rb") as file_in:
        prime_dict = pickle.load(file_in)

    # Precompute flat arrays of log(prime)/sqrt(power) (amplitude)
    # and log(power) (cosine frequency) for all prime powers
    amplitudes = []
    log_powers = []
    for prime, powers in prime_dict.items():
        for power in powers:
            amplitudes.append(np.log(prime) / np.sqrt(power))
            log_powers.append(np.log(power))
    amplitudes = np.array(amplitudes)
    log_powers = np.array(log_powers)

    x = np.linspace(0, 100, 1_000)
    y = F(x, amplitudes, log_powers)

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

    plt.axvline(
        x=mpmath.zetazero(1).imag,
        ymax=0.9,
        linestyle="--",
        color="red",
        label="Zeta Zero",
    )
    for i in range(2, 30):
        plt.axvline(x=mpmath.zetazero(i).imag, ymax=0.9, linestyle="--", color="red")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
