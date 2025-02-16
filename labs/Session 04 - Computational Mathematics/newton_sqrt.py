#!/usr/bin/env python3
"""newton_sqrt.py"""

import numpy as np


def sqrt(x):
    low, high = 0, x
    est = (low + high) / 2
    while np.abs(est**2 - x) > 1e-10:
        if est**2 < x:
            low = est
        else:
            high = est
        est = (low + high) / 2
    return est


def main():
    x = 168923.74
    print(f"x = {x}")
    print(f"sqrt(x) = {np.sqrt(x)}")


if __name__ == "__main__":
    main()
