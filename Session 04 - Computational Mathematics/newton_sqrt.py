#!/usr/bin/env -S uv run
"""newton_sqrt.py"""

import numpy as np


def newton_sqrt(x):
    if not np.isfinite(x) or x < 0:
        raise ValueError("Expected a finite nonnegative number")
    if x == 0:
        return 0.0
    low, high = 0.0, max(1.0, x)
    est = (low + high) / 2
    while not np.isclose(est, x / est, rtol=1e-15, atol=0.0):
        if est < x / est:
            low = est
        else:
            high = est
        next_est = low + (high - low) / 2
        if next_est == est:
            break
        est = next_est
    return est


def main():
    x = 168923.74
    print(f"x = {x}")
    print(f"Bisection sqrt(x) = {newton_sqrt(x):.12f}")
    print(f"Numpy  sqrt(x) = {np.sqrt(x):.12f}")


if __name__ == "__main__":
    main()
