#!/usr/bin/env python3
"""perfect_numbers.py"""

import numpy as np


def is_perfect(n):
    x = np.arange(1, n)
    factors = x[np.where(n % x == 0)]
    return np.sum(factors) == n


def main():
    for n in range(2, 10_000):
        if is_perfect(n):
            print(f"{n:,} is a perfect number")


if __name__ == "__main__":
    main()
