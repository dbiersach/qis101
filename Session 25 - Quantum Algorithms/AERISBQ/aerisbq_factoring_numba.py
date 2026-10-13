#!/usr/bin/env -S uv run
"""aerisbq_factoring_numba.py"""

# A numba-accelerated solver for the AERISBQ method. The algorithm is identical to
# aerisbq_factoring.py -- the bit-by-bit binary-quadratic-form search -- but the
# whole hot loop compiles to machine code, so it runs the same factorizations far
# faster. Two things are given up for that speed:
#
#   * Fixed 64-bit integers instead of Python's arbitrary precision, so N must be
#     an odd semiprime below 2^62. aerisbq_factoring.py has no such ceiling.
#   * No show_forms / trace display. This file is the fast solver only; read
#     aerisbq_factoring.py for the readable, step-by-step account of the method.
#
# Everything -- expanding each form, the four-step sieve, both divisor tests, the
# round-end probe, and the sqrt(N) residue test -- lives inside one @njit function
# working on int64 scalars and NumPy arrays. gcd, isqrt, and the dedup set are
# written out longhand because numba's nopython mode does not take the library
# versions.

import time

import numpy as np
from numba import njit

# Each surviving form's residues r, s are below sqrt(N) < 2^31 for N < 2^62, so a
# pair packs into one int64 dedup key as  min * 2^31 + max.
PACK = np.int64(1) << 31
LIMIT = 1 << 62  # int64 arithmetic stays exact for N below this


# The @njit helpers below carry no type annotations: numba types them from the
# int64 values it is handed, and a Python `int` hint clashes with those numpy
# scalars under a static type checker. The pure-Python functions keep their hints.


@njit(cache=True)
def _gcd(a, b):
    """Euclid's algorithm on the magnitudes of a and b."""
    if a < 0:
        a = -a
    if b < 0:
        b = -b
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def _isqrt(n):
    """Integer square root: floor(sqrt(n))."""
    if n < 2:
        return n
    x = np.int64(np.sqrt(np.float64(n)))
    while x * x > n:
        x -= 1
    while (x + 1) * (x + 1) <= n:
        x += 1
    return x


@njit(cache=True)
def _analyze(a, r, s, N):
    """Four-step sieve on (a*x + r)(a*y + s) = N (see aerisbq_factoring.analyze_form).

    Returns ``(survives, factor)``; factor == 0 means the form survives but reveals
    no divisor here.
    """
    num = N - r * s
    if num % a != 0:
        return False, 0  # trivial rejection
    n2 = num // a
    g = _gcd(_gcd(_gcd(a, s), r), n2)
    if g == 0:
        g = 1
    a2 = a // g
    b2 = s // g
    c2 = r // g
    n2 = n2 // g

    a3 = -a2
    b3 = a2
    c3 = -2 * (b2 - c2)
    d3 = 2 * (b2 + c2)
    n3 = 4 * n2
    g = _gcd(_gcd(_gcd(_gcd(a3, b3), c3), d3), n3)
    if g == 0:
        g = 1
    a3 //= g
    c3 //= g
    d3 //= g
    n3 //= g

    a4 = 4 * a3
    b4 = -4 * a3
    c4 = -2 * c3
    d4 = c3 + d3
    n4 = n3
    g = _gcd(_gcd(_gcd(_gcd(a4, b4), c4), d4), n4)
    if g == 0:
        g = 1
    a4 //= g
    b4 //= g
    c4 //= g
    d4 //= g
    n4 //= g

    g = _gcd(n4, _gcd(a4, c4))
    if g > 1:
        a4 //= g
        c4 //= g
        n4 //= g
    if n4 < 0:
        return False, 0  # residue has overshot the true factor

    # The (1,1) test gives the factor a + r; the n4 test gives the factor n4.
    if a4 + b4 + c4 + d4 == n4 and 1 < a + r < N and N % (a + r) == 0:
        return True, a + r
    if 1 < n4 < N and N % n4 == 0:
        return True, n4
    return True, 0


@njit(cache=True)
def _factor_core(N):
    """Run the whole AERISBQ search on N. Returns a nontrivial divisor, or 0."""
    if N % 2 == 0:
        return 2
    root = _isqrt(N)

    r_live = np.empty(1, np.int64)
    s_live = np.empty(1, np.int64)
    r_live[0] = 1
    s_live[0] = 1  # root form (2x + 1)(2y + 1) = N
    a = np.int64(2)

    max_rounds = 0
    t = N
    while t:
        t >>= 1
        max_rounds += 1

    for _round in range(1, max_rounds + 1):
        na = 2 * a
        r_child = np.empty(4 * len(r_live), np.int64)
        s_child = np.empty(4 * len(r_live), np.int64)
        seen = set()
        cnt = 0
        for i in range(len(r_live)):
            r = r_live[i]
            s = s_live[i]
            for j in range(4):
                if j == 0:
                    rp = r
                    sp = s
                elif j == 1:
                    rp = r + a
                    sp = s + a
                elif j == 2:
                    rp = r
                    sp = s + a
                else:
                    rp = r + a
                    sp = s
                # Past sqrt(N) the smaller factor equals a residue; test directly.
                if na > root:
                    if 1 < rp < N and N % rp == 0:
                        return rp
                    if 1 < sp < N and N % sp == 0:
                        return sp
                    continue
                survives, fac = _analyze(na, rp, sp, N)
                if not survives:
                    continue
                if fac != 0:
                    return fac
                key = min(rp, sp) * PACK + max(rp, sp)  # collapse (r, s) with (s, r)
                if key in seen:
                    continue
                seen.add(key)
                r_child[cnt] = rp
                s_child[cnt] = sp
                cnt += 1

        # Round-end probe: a factor is a residue plus a small multiple of na.
        for i in range(cnt):
            rp = r_child[i]
            sp = s_child[i]
            for cand in (
                rp + na,
                rp + 2 * na,
                rp + 4 * na,
                rp + 8 * na,
                sp + na,
                sp + 2 * na,
                sp + 4 * na,
                sp + 8 * na,
            ):
                if 1 < cand < N and N % cand == 0:
                    return cand

        if cnt == 0:
            break
        r_live = r_child[:cnt].copy()
        s_live = s_child[:cnt].copy()
        a = na

    return 0


def factor(N: int) -> tuple[int, int] | None:
    """Factor an odd semiprime N (below 2^62) with the numba AERISBQ solver.

    Parameters
    ----------
    N : int
        A composite that is the product of exactly two odd primes, 1 < p < q, and
        below 2^62 so the search stays within exact 64-bit arithmetic.

    Returns
    -------
    tuple[int, int] | None
        ``(p, q)`` with p < q if a factorization is found, else None.
    """
    if N < 4:
        return None
    if N >= LIMIT:
        raise ValueError(
            f"this numba solver handles N below 2^62; "
            f"use aerisbq_factoring.py for N = {N}"
        )
    divisor = int(_factor_core(N))
    if divisor <= 1 or divisor >= N or N % divisor != 0:
        return None
    p, q = divisor, N // divisor
    return (min(p, q), max(p, q))


def main() -> None:
    factor(15)  # warm up the JIT compiler so it is excluded from the timings

    print("=== AERISBQ factoring (numba) ===\n")
    for N in (
        26329,
        124169,
        20082467,
        100003 * 100019,
        4087,
        143,
        3599,
        10403,
        799819,
        6422309,
    ):
        start_time = time.perf_counter()
        result = factor(N)
        elapsed_time = time.perf_counter() - start_time
        assert result is not None and result[0] * result[1] == N
        print(f"{N} = {result[0]} x {result[1]}   (run time: {elapsed_time:.3f} sec)")

    # The balanced 56-bit case from aerisbq_factoring.py, where the pure-Python
    # solver runs for over two minutes.
    print("\n=== Balanced 56-bit semiprime ===")
    N = 52000010780000363  # 200000033 x 260000011
    start_time = time.perf_counter()
    result = factor(N)
    elapsed_time = time.perf_counter() - start_time
    assert result is not None and result[0] * result[1] == N
    print(f"  {N} ({N.bit_length()}-bit) = {result[0]} x {result[1]}")
    print(f"  (run time: {elapsed_time:.3f} sec)")


if __name__ == "__main__":
    main()
