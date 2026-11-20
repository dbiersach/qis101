#!/usr/bin/env -S uv run
"""herons_method.py"""

import numpy as np


def newton_sqrt(x: float) -> float:
    """Bisect until estimate and x / estimate agree relatively or rounding stalls."""
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
        next_est = (low + high) / 2
        if next_est == est:
            break
        est = next_est
    return est


def heron_sqrt(s: float) -> float:
    """Average until x and s / x agree relatively or rounding stalls."""
    if not np.isfinite(s) or s < 0:
        raise ValueError("Expected a finite nonnegative number")
    if s == 0:
        return 0.0
    x = max(1.0, s / 2)
    while not np.isclose(x, s / x, rtol=1e-15, atol=0.0):
        next_x = (s / x + x) / 2
        if next_x == x:
            break
        x = next_x
    return x


def main():
    x = 168923.74
    print(f"x = {x}")
    print(f"Bisection sqrt(x) = {newton_sqrt(x):.16g}")
    print(f"Numpy  sqrt(x)    = {np.sqrt(x):.16g}")
    print(f"Heron  sqrt(x)    = {heron_sqrt(x):.16g}")


if __name__ == "__main__":
    main()
