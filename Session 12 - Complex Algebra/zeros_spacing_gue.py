#!/usr/bin/env -S uv run
"""zeros_spacing_gue.py"""

import lzma
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

zeta_zeros = {}


def generate_gue_eigenvalues(n):
    # Diagonal: real Gaussian entries
    diag = np.random.normal(loc=0.0, scale=1.0, size=n)

    # Upper triangle (excluding diagonal): complex Gaussian
    real_part = np.random.normal(0, np.sqrt(0.5), size=(n, n))
    imag_part = np.random.normal(0, np.sqrt(0.5), size=(n, n))
    upper = real_part + 1j * imag_part

    # Create Hermitian matrix
    H = np.triu(upper, k=1)  # upper triangle (excluding diagonal)
    H += H.conj().T  # add transpose to get Hermitian matrix
    np.fill_diagonal(H, diag)  # set diagonal separately

    # Compute and return eigenvalues
    eigenvalues = sorted(np.linalg.eigvalsh(H))
    eigenvalues += eigenvalues[-1]
    eigenvalues /= 4 * np.sqrt(n)
    return eigenvalues


def main():
    # Read pickle file containing the zeta zeros
    file_path = Path(__file__).parent / "zeta_zeros.pickle.xz"
    with lzma.open(file_path, "rb") as file_in:
        global zeta_zeros
        zeta_zeros = pickle.load(file_in)
        # Take only the first 1000 zeta zeros
        zeta_zeros = zeta_zeros[:1000]

    # Create a 1d array of 1000 x 1000 GUE eigenvalues
    eigenvals_gue = generate_gue_eigenvalues(1000)

    # Scale the eigenvalues to span the range of Zeta Zeros
    eigenvals_gue *= zeta_zeros[-1] - zeta_zeros[0]

    # Plot the Zeta Zeros in 10 lines of 100 values each
    once = True
    plt.figure(Path(__file__).name)
    for i in range(10):
        start = i * 100
        stop = start + 99
        x = zeta_zeros[start:stop]
        x = x - x[0]
        y = np.full_like(x, i * 10 + 2)
        if once:
            plt.scatter(x[0], y[0], c="b", s=3, label="Zeta Zeros")
            once = False
        plt.scatter(x, y, c="b", s=3)

    # Plot the GUE eigenvalues in 10 lines of 100 values each
    once = True
    for i in range(10):
        start = i * 100
        stop = start + 99
        x = eigenvals_gue[start:stop]
        x = x - x[0]
        y = np.full_like(x, i * 10 + 4)
        if once:
            plt.scatter(x[0], y[0], c="r", s=3, label="GUE Eigenvalues")
            once = False
        plt.scatter(x, y, c="r", s=3)

    plt.title("Zeta Zero Spacing vs. GUE Eigenvalues")
    plt.xlabel("Imaginary Component of each Zeta Root")
    plt.ylabel("100 spacings per row")
    plt.xlim(0, 100)
    plt.xticks([])
    plt.legend(loc="upper right", framealpha=1.0, facecolor="white")
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.show()


if __name__ == "__main__":
    main()
