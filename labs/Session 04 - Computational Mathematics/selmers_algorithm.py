#!/usr/bin/env python3
"""selmers_algorithm.py"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def frobenius_number(coprime_triplet: tuple[int, int, int]) -> int:
    """
    Computes the Frobenius number for three pairwise coprime integers using Selmer's algorithm.
    """

    # Unpack the passed in tuple
    a, b, c = coprime_triplet

    # Compute the Frobenius number for (a, b)
    g_ab = a * b - a - b  # Classical two-number Frobenius formula

    # Track which residues mod c are representable
    reachable = np.zeros(c, dtype=np.bool_)

    # Generate reachable numbers using (a, b)
    for x in prange(b):
        for y in prange(a):
            num = x * a + y * b
            reachable[num % c] = True  # Mark reachable residues mod c

    # Find the largest non-representable number
    max_non_representable = -1

    for n in range(g_ab, -1, -1):
        if not reachable[n % c]:  # If this residue is missing, it's non-representable
            max_non_representable = n
            break

    return max_non_representable


def main():
    triplet = (15, 47, 997)  # g() = 643
    # triplet = (15, 997, 7919)  # g() = 13943
    # triplet = (7919, 12553, 17389)  # g() = 2711658
    print(f"g{triplet} = {frobenius_number(triplet)}")


if __name__ == "__main__":
    main()
