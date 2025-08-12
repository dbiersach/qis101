#!/usr/bin/env python3
"""frobenius_parallel.py"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def frobenius_number(coprime_triplet: tuple[int, int, int]) -> int:
    a, b, c = coprime_triplet
    # Define the upper bound using the rule of thumb
    max_n = min(2 * a * b * c - a * b - a * c - b * c, (a * b * c) // 2)
    # Allocate array of Booleans to represent if an integer is representable
    representables = np.zeros(max_n + 1, dtype=np.bool_)
    # Mark which integers are representable as a linear combination of a, b, c
    for i in prange(max_n // a + 1):
        n1 = a * i
        for j in range(max_n // b + 1):
            n2 = n1 + b * j
            if n2 > max_n:
                break
            for k in range(max_n // c + 1):
                n3 = n2 + c * k
                if n3 > max_n:
                    break
                representables[n3] = True
    # Return the last non-representable number
    return np.flatnonzero(~representables)[-1]


def main():
    triplet = (15, 47, 997)  # g() = 643
    print(f"g{triplet} = {frobenius_number(triplet)}")


if __name__ == "__main__":
    main()
