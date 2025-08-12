#!/usr/bin/env python3
"""collatz_conjecture.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def stop_time(n):
    c = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        c = c + 1
    return c


def main():
    max_n = 1_000_000
    x = np.arange(max_n)
    f = np.vectorize(stop_time)
    y = f(x)
    plt.figure(Path(__file__).name)
    plt.hist(y, bins=500, color="blue")
    plt.title(f"Collatz Conjecture (n < {max_n:,})")
    plt.xlabel("Stopping Time")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main()
