#!/usr/bin/env -S uv run
"""lattice_circle.py"""

import numpy as np
from numba import njit


@njit
def lattice_points(r):
    c = 0
    for x in range(-r, r + 1):
        for y in range(-r, r + 1):
            if x**2 + y**2 <= r**2:
                c = c + 1
    return c


def main():
    for radius in np.linspace(1000, 10000, 10):
        act = lattice_points(radius)
        est = int(np.pi * radius**2 + 2 * np.sqrt(2 * np.pi * radius))
        err = (est - act) / act
        print(
            f"Radius = {int(radius):>6,}"
            f"  Act Points = {act:>12,}"
            f"  Est Points = {est:>12,}"
            f"  % Rel. Err = {err:0.4%}"
        )


if __name__ == "__main__":
    main()
