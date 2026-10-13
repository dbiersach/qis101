#!/usr/bin/env -S uv run
"""create_semiprimes.py"""

# Build a table of test semiprimes for the AERISBQ-versus-Fermat timing study.
# For every bit length from 10 to 60, we pick 20 semiprimes N = p * q whose two
# primes range from a LARGE gap (p only a few bits, q huge) down to a SMALL gap
# (p and q both near sqrt(N)). That spread matters because AERISBQ and Fermat are
# hard on opposite ends of it: AERISBQ's cost grows with the smaller prime (worst
# when balanced), Fermat's grows with (sqrt(q) - sqrt(p))^2 (worst when lopsided).
#
# The CSV has three columns -- N, p, q -- with the rows for each bit length
# ordered from the largest prime gap to the smallest.

import csv
import random
from pathlib import Path
from typing import cast

import sympy

MIN_BITS = 10
MAX_BITS = 60
PER_BITLEN = 20


def make_semiprime(bits: int, p_bits: int) -> tuple[int, int, int] | None:
    """Return (N, p, q) with N exactly `bits` bits and p about `p_bits` bits.

    p and q are distinct odd primes with p <= q. Returns None if this draw does
    not land N in the target bit-length window (the caller just tries again).
    """
    lo_n = 1 << (bits - 1)
    hi_n = 1 << bits
    # sympy ships no type stubs, so cast the (runtime) int back to a typed int.
    p = cast("int", sympy.randprime(1 << (p_bits - 1), 1 << p_bits))
    q_lo = lo_n // p + 1
    q_hi = hi_n // p
    if q_hi <= q_lo:
        return None
    try:
        q = cast("int", sympy.randprime(q_lo, q_hi))
    except ValueError:
        return None
    if q == p:
        return None
    n = p * q
    if n.bit_length() != bits:
        return None
    return (n, min(p, q), max(p, q))


def semiprimes_for_bits(bits: int) -> list[tuple[int, int, int]]:
    """Collect PER_BITLEN distinct semiprimes spanning the prime-gap spectrum."""
    # The smallest prime is 3 bits (p >= 5). A 2-bit prime is degenerate for this
    # comparison: p = 2 makes N even, and p = 3 sits below AERISBQ's first place
    # value (4), so its early tests miss it and the search grinds to sqrt(N) --
    # a boundary artifact, not the method's true cost. Every p >= 5 is caught
    # cleanly, so the work curve rises smoothly.
    min_p_bits = 3
    max_p_bits = max(min_p_bits, bits // 2)
    seen: set[int] = set()
    collected: list[tuple[int, int, int]] = []
    for i in range(PER_BITLEN):
        # Walk the target prime size from 3 bits (lopsided) to bits//2 (balanced).
        frac = i / (PER_BITLEN - 1)
        target = min_p_bits + round((max_p_bits - min_p_bits) * frac)
        result = None
        for _ in range(500):
            result = make_semiprime(bits, target)
            if result is not None and result[0] not in seen:
                break
            result = None
        if result is None:  # rare fallback: any distinct semiprime of this size
            for _ in range(5000):
                cand = make_semiprime(bits, random.randint(min_p_bits, max_p_bits))
                if cand is not None and cand[0] not in seen:
                    result = cand
                    break
        if result is not None:
            seen.add(result[0])
            collected.append(result)
    collected.sort(key=lambda t: t[2] - t[1], reverse=True)  # largest gap first
    return collected


def main() -> None:
    random.seed(2024)
    rows: list[tuple[int, int, int]] = []
    for bits in range(MIN_BITS, MAX_BITS + 1):
        batch = semiprimes_for_bits(bits)
        rows.extend(batch)
        print(f"  {bits:>2}-bit: {len(batch)} semiprimes")

    file_path = Path(__file__).parent / "semiprimes.csv"
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "p", "q"])
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} semiprimes to {file_path}")


if __name__ == "__main__":
    main()
