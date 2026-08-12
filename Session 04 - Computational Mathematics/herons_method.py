#!/usr/bin/env -S uv run
"""herons_method.py"""

import numpy as np


def newton_sqrt(x):
    low, high = 0, x
    est = (low + high) / 2
    while np.abs(est**2 - x) > 1e-10:
        if est**2 < x:
            low = est
        else:
            high = est
        est = (low + high) / 2
    return est


def heron_sqrt(s):
    x = s / 2
    while x**2 - s > 1e-10:
        x = (s / x + x) / 2
    return x


def main():
    x = 168923.74
    print(f"x = {x}")
    print(f"Newton sqrt(x) = {newton_sqrt(x):.12f}")
    print(f"Numpy  sqrt(x) = {np.sqrt(x):.12f}")
    print(f"Heron  sqrt(x) = {heron_sqrt(x):.12f}")


if __name__ == "__main__":
    main()
