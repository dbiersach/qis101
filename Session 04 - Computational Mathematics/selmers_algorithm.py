#!/usr/bin/env -S uv run
"""selmers_algorithm.py"""

import numpy as np
from numba import njit, prange


@njit
def modular_inverse(value: int, n: int) -> int:
    """Find inverse such that (value * inverse) = 1 mod n

    Example: The inverse of 3 mod 7 is 5 because (3 * 5) mod 7 = 1
    """
    old_r = value % n
    r = n
    old_s = 1
    s = 0
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    if old_r != 1:
        return -1
    return old_s % n


@njit(parallel=True)
def apery_set_mod_a(a: int, b: int, c: int) -> np.ndarray:
    """Return the Apery set with respect to a for the pair (b, c)"""
    initial_best = np.iinfo(np.int64).max
    inverse_c = modular_inverse(c, a)
    apery = np.empty(a, dtype=np.int64)
    for residue in prange(a):
        best = initial_best
        for y in range(a):
            remaining = (residue - b * y) % a
            z = (remaining * inverse_c) % a
            value = b * y + c * z
            if value < best:
                best = value
        apery[residue] = best
    return apery


def frobenius_number(triplet: tuple[int, int, int]) -> int:
    """Return the Frobenius number of a pairwise-coprime triplet"""
    a, b, c = triplet
    apery = apery_set_mod_a(a, b, c)
    return int(np.max(apery) - a)


def main() -> None:
    # a must be the smallest #, and gcd(a,b,c) must be = 1
    triplet = (7919, 12553, 17389)  # g() = 2711658
    print(f"g{triplet} = {frobenius_number(triplet)}")


if __name__ == "__main__":
    main()
