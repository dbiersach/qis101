#!/usr/bin/env -S uv run
"""selmers_algorithm.py"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def frobenius_number(triplet, limit):
    """
    Uses Selmer-style dynamic marking to find the Frobenius number.
    All combinations ax + by + cz are marked as representable.
    """
    a, b, c = triplet
    reachable = np.zeros(limit + 1, dtype=np.uint8)

    # Parallel loop for performance
    for x in prange(limit // a + 1):
        ax = x * a
        for y in range((limit - ax) // b + 1):
            axy = ax + y * b
            for z in range((limit - axy) // c + 1):
                val = axy + z * c
                if val <= limit:
                    reachable[val] = 1

    # Find the largest number that is not reachable
    for i in range(limit, -1, -1):
        if reachable[i] == 0:
            return i
    return -1  # This should never happen if limit is high enough


def main():
    triplet = (7919, 12553, 17389)  # g() = 2711658
    limit = 3_000_000
    print(f"g{triplet} = {frobenius_number(triplet, limit)}")


if __name__ == "__main__":
    main()
