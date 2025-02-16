#!/usr/bin/env python3
"""plot_multigraphs.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import plot_parabola
import plot_polynomial
import plot_rings
import plot_rose_curves


def main():
    plt.figure(Path(__file__).name, figsize=(12, 8))
    plot_parabola.plot(plt.subplot(221))
    plot_polynomial.plot(plt.subplot(222))
    plot_rings.plot(plt.subplot(223))
    plot_rose_curves.plot(plt.subplot(224, projection="polar"))
    plt.show()


if __name__ == "__main__":
    main()
