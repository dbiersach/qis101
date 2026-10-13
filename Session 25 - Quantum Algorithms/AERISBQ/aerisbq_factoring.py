#!/usr/bin/env -S uv run
"""aerisbq_factoring.py"""

# AERISBQ = Algorithmic Entropy Reduction in Sequenced Binary Quadratic (forms).
#
# A classical (non-quantum) method for factoring a composite N known to be the
# product of exactly two odd primes, 1 < p < q (an RSA-type semiprime). It
# recovers the two factors one binary bit at a time.
#
# ---------------------------------------------------------------------------
# The idea
# ---------------------------------------------------------------------------
# Multiplying p by q is a lossy, one-way "compression": from N alone we cannot
# read back which two numbers were multiplied. AERISBQ recovers the factors from
# the least significant bit (LSB) upward, keeping only the bit-combinations that
# could still multiply out to N. Each surviving guess is a binary quadratic form
#
#       (a * x + r) * (a * y + s) = N
#
# where a = 2^k is the place value of the bit being decided and r, s are the low
# bits of p and q pinned down so far (p = r mod a, q = s mod a). We never solve a
# form exactly -- there is no fast general method for that -- we only ask whether
# it could have a positive-integer solution and drop the ones that cannot.
#
# ---------------------------------------------------------------------------
# One round decides the next bit of p and of q
# ---------------------------------------------------------------------------
# Each surviving form spawns up to four children (the four states of the next
# bit-pair). A child doubles the place value, a -> 2a, and runs a four-step sieve:
#
#   Step 1  Expand the product. Every term except r*s is a multiple of a, so a
#           child can have a solution only when N == r*s (mod a); otherwise it is
#           a "trivial rejection". Dividing by a leaves a*xy + s*x + r*y = (N-r*s)/a.
#   Step 2  u-v substitution  x = (v-u)/2, y = (v+u)/2, turning the xy term into
#           squares and forcing u and v to share the same parity.
#   Step 3  lambda-substitution  u = v - 2l, which cancels the v^2 term and leaves
#           a form with no y^2 term.
#   Step 4  With no y^2 term, isolate y and GCD-reduce. A factor shows up two ways:
#           the (1,1) test  a4 + b4 + c4 + d4 == n4  yields the factor a + r, and
#           whenever the reduced right-hand side n4 divides N, n4 is itself a factor.
#
# A form (r, s) and its swap (s, r) are the same equation, so they collapse to one
# form and the round carries half as many. The (1,1) test is asymmetric in r and s
# while the n4 test is symmetric, so between them a factor is caught in whichever
# orientation survives the collapse.
#
# ---------------------------------------------------------------------------
# Two shortcuts that reach a factor early
# ---------------------------------------------------------------------------
# At each round's end a divisor probe tries each residue plus the next four place
# values -- r and s each plus 2^(n+1), 2^(n+2), 2^(n+3), 2^(n+4) -- as a divisor of
# N. This catches a factor whose remaining high bits are a single 1 in the next
# four positions, up to three rounds before Step 4 reaches it.
#
# And once the place value passes sqrt(N), the smaller factor is below it and so
# must equal one of the residues outright, so the residues are tested directly.
#
# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
# This is a research-grade method, not a practical attack on real RSA. The search
# reaches a factor around the most-significant-bit round of the smaller factor, so
# it costs on the order of sqrt(N) forms -- the same ballpark as trial division,
# and expensive past N ~ 10^12, worst when p and q are near-twin. No cheaper test
# prunes the surviving forms: they are the square roots of N modulo 2^k, and
# separating the true pair from the rest is the factoring problem itself. It works
# only for an odd semiprime; the (2x+1)(2y+1) root assumes both factors are odd.

import time
from math import gcd, isqrt


def reduce_terms(*coeffs: int) -> tuple[int, ...]:
    """Divide integer coefficients by their common GCD.

    The GCD is taken over absolute values, so signs are preserved. All-zero input
    is returned unchanged (the GCD is treated as 1).
    """
    g = 0
    for c in coeffs:
        g = gcd(g, abs(c))
    if g == 0:
        return coeffs
    return tuple(c // g for c in coeffs)


def four_step_form(
    a: int, r: int, s: int, N: int
) -> tuple[int, int, int, int, int] | None:
    """Run Steps 1-4 of the sieve and return the reduced Step-4 coefficients.

    The Step-4 form is  a4*x^2 + b4*xy + c4*x + d4*y = n4, and this returns those
    five integers. It returns None for a trivial rejection (N != r*s mod a) -- the
    one case with no Step-4 form, because Step 1 already has no integer right side.

    Every division below is exact (the divisor is a true common factor), so
    ordinary integer division is safe and no rounding is involved.
    """
    # Step 1: expand and demand an integer right-hand side. All terms except r*s
    # are multiples of a, so a solution needs N == r*s (mod a).
    num = N - r * s
    if num % a != 0:
        return None  # trivial rejection
    # Reduced Step-1 form  a2*xy + b2*x + c2*y = n2, with b2 = s and c2 = r.
    a2, b2, c2, n2 = reduce_terms(a, s, r, num // a)

    # Step 2: u-v substitution  ->  a3*u^2 + b3*v^2 + c3*u + d3*v = n3.
    a3, b3, c3, d3, n3 = reduce_terms(-a2, a2, -2 * (b2 - c2), 2 * (b2 + c2), 4 * n2)

    # Step 3: lambda-substitution. The y^2 coefficient is a3 + b3 = -a2 + a2 = 0,
    # so it drops out, leaving  a4*x^2 + b4*xy + c4*x + d4*y = n4.
    assert a3 + b3 == 0  # the v^2 term cancels by construction
    a4, b4, c4, d4, n4 = reduce_terms(4 * a3, -4 * a3, -2 * c3, c3 + d3, n3)

    # Step 4: GCD-reduce the x^2, x, and constant terms.
    g = gcd(n4, gcd(abs(a4), abs(c4)))
    if g > 1:
        a4 //= g
        c4 //= g
        n4 //= g
    return (a4, b4, c4, d4, n4)


def analyze_form(a: int, r: int, s: int, N: int) -> tuple[bool, int | None]:
    """Run the four-step AERISBQ sieve on the form (a*x + r)(a*y + s) = N.

    Parameters
    ----------
    a : int
        Current place value, a power of two.
    r, s : int
        Low bits of p and q pinned down so far (0 <= r, s < a).
    N : int
        The composite being factored.

    Returns
    -------
    tuple[bool, int | None]
        ``(survives, factor)``. ``survives`` is False for a form that cannot have
        a positive-integer solution (a trivial rejection, or one driven negative
        by Step 4). ``factor`` is a nontrivial divisor of N when this form reveals
        one, by either test below, else None.
    """
    coeffs = four_step_form(a, r, s, N)
    if coeffs is None:
        return (False, None)  # trivial rejection: no integer right-hand side
    a4, b4, c4, d4, n4 = coeffs
    if n4 < 0:
        return (False, None)  # residue has overshot the true factor; prune

    # The (1,1) test: the reduced form has a solution at (x, y) = (1, 1) exactly
    # when its coefficients sum to n4, giving the factor a + r. This test is
    # asymmetric in r and s.
    if a4 + b4 + c4 + d4 == n4 and 1 < a + r < N and N % (a + r) == 0:
        return (True, a + r)

    # The n4 test: n4 lands on the cofactor once it is fully pinned down, so
    # whenever n4 divides N it is a factor. This test is symmetric in r and s
    # (n4 comes from N - r*s), so it fires for either orientation.
    if 1 < n4 < N and N % n4 == 0:
        return (True, n4)

    return (True, None)


def next_bit_forms(a: int, r: int, s: int) -> list[tuple[int, int, int]]:
    """Return the four child forms for the next bit of p and q.

    Doubling the place value from ``a`` to ``2a`` exposes one new bit in each of p
    and q; the four (bit_p, bit_q) combinations give the four ``(2a, r', s')``
    candidate forms, before any sieving.
    """
    new_a = 2 * a
    return [(new_a, r + b, s + c) for b, c in ((0, 0), (a, a), (0, a), (a, 0))]


def format_form(a: int, r: int, s: int, N: int) -> str:
    """Render one BQF as the product string ``(a*x + r)(a*y + s) = N``."""
    return f"({a}x+{r})({a}y+{s})={N}"


def format_step4(a4: int, b4: int, c4: int, d4: int, n4: int) -> str:
    """Render the Step-4 form  a4*x^2 + b4*xy + c4*x + d4*y = n4  compactly.

    Zero terms are dropped and a coefficient of +-1 shows only its variable, so
    the coefficients (-2, 4, 0, 1, 3291) print as ``-2x^2+4xy+y=3291``.
    """
    out = ""
    for coeff, var in ((a4, "x^2"), (b4, "xy"), (c4, "x"), (d4, "y")):
        if coeff == 0:
            continue
        sign = "-" if coeff < 0 else ("+" if out else "")
        magnitude = abs(coeff)
        body = var if magnitude == 1 else f"{magnitude}{var}"
        out += f"{sign}{body}"
    if not out:  # every term vanished (a degenerate form); show a bare 0
        out = "0"
    return f"{out}={n4}"


def step4_solutions(
    a4: int, b4: int, c4: int, d4: int, n4: int
) -> list[tuple[int, int]]:
    """Return every integer (x, y) solving a4*x^2 + b4*xy + c4*x + d4*y = n4.

    The form is linear in y, so for each integer x the value of y is forced:

        y = (n4 - a4*x^2 - c4*x) / (b4*x + d4)

    and y is an integer exactly when (b4*x + d4) divides that numerator. A short
    calculation shows this is the same as (b4*x + d4) dividing the fixed constant

        K = b4^2 * n4 + b4*c4*d4 - a4*d4^2

    so the solutions read straight off the divisors D of K: keep the ones with
    D == d4 (mod b4), then x = (D - d4) / b4 and y = numerator / D.
    """
    if b4 == 0:
        return []
    K = b4 * b4 * n4 + b4 * c4 * d4 - a4 * d4 * d4
    if K == 0:
        return []  # degenerate: the divisor argument does not pin x down
    magnitude = abs(K)
    sols: set[tuple[int, int]] = set()
    d = 1
    while d * d <= magnitude:
        if magnitude % d == 0:
            for divisor in (d, -d, magnitude // d, -(magnitude // d)):
                if (divisor - d4) % b4 != 0:
                    continue
                x = (divisor - d4) // b4
                num = n4 - a4 * x * x - c4 * x
                if num % divisor != 0:
                    continue
                y = num // divisor
                if a4 * x * x + b4 * x * y + c4 * x + d4 * y == n4:
                    sols.add((x, y))
        d += 1
    return sorted(sols)


def step4_witness(
    a4: int, b4: int, c4: int, d4: int, n4: int
) -> tuple[int, int, bool] | None:
    """Pick one solution to display and report whether a positive one exists.

    Returns ``(x, y, is_good)``, where ``is_good`` is True when the form has a
    solution in positive integers (x >= 1, y >= 1). The chosen (x, y) is the
    solution at the smallest nontrivial |b4*x + d4| -- the one tied to the
    smallest factor -- drawn from the positive solutions when any exist, else from
    all solutions. Returns None when the form has no integer solution at all.
    """
    sols = step4_solutions(a4, b4, c4, d4, n4)
    if not sols:
        return None
    is_good = any(x >= 1 and y >= 1 for x, y in sols)
    pool = [(x, y) for x, y in sols if x >= 1 and y >= 1] if is_good else sols

    def rank(sol: tuple[int, int]) -> tuple[bool, int, int]:
        divisor = abs(b4 * sol[0] + d4)
        return (divisor <= 1, divisor, sol[0])  # nontrivial |D| first, then smaller

    x, y = min(pool, key=rank)
    return (x, y, is_good)


def format_form_line(a: int, r: int, s: int, N: int) -> str:
    """Build one show_forms line for the form (a*x + r)(a*y + s) = N.

    The line is the product form, then its Step-4 form, then the first (x, y) that
    satisfies that Step-4 form, then the word ``Good`` when the form has a
    positive-integer solution.
    """
    product = format_form(a, r, s, N)
    coeffs = four_step_form(a, r, s, N)
    if coeffs is None:
        return f"{product:<24}(trivial rejection)"
    bqf = format_step4(*coeffs)
    witness = step4_witness(*coeffs)
    if witness is None:
        return f"{product:<24}{bqf}"
    x, y, is_good = witness
    solution = f"(x, y) = ({x}, {y})"
    tag = "Good" if is_good else ""
    return f"{product:<24}{bqf:<24}{solution:<26}{tag}".rstrip()


def factor(
    N: int, *, show_forms: bool = False, trace: bool = False
) -> tuple[int, int] | None:
    """Factor an odd semiprime N = p*q with the AERISBQ method.

    Parameters
    ----------
    N : int
        A composite known to be the product of exactly two odd primes, 1 < p < q.
    show_forms : bool, optional
        When True, print every surviving BQF in each round: the product form
        ``(a*x + r)(a*y + s) = N``, its reduced Step-4 form, the first (x, y) that
        satisfies the Step-4 form, and ``Good`` when that form has a positive-
        integer solution. Meant for small N; the list grows each round.
    trace : bool, optional
        When True, print the number of live forms in each round -- a compact view
        of how the search grows and then collapses to a factor.

    Returns
    -------
    tuple[int, int] | None
        ``(p, q)`` with p < q if a factorization is found, else None.
    """
    if N % 2 == 0:
        return (2, N // 2)  # even N: the (2x+1)(2y+1) root does not apply

    def found(divisor: int) -> tuple[int, int]:
        """Normalize a discovered divisor to (p, q), print, and return it."""
        p, q = min(divisor, N // divisor), max(divisor, N // divisor)
        if show_forms:
            print(f"  round {round_num:>2}: factor found  ->  {N} = {p} x {q}")
        if trace:
            print(
                f"  round {round_num:>2}: factor found after testing {forms_tested} forms"
            )
            print(f"  {N} = {p} x {q}")
        return (p, q)

    root = isqrt(N)  # once the place value passes this, the small factor is fixed
    live: list[tuple[int, int, int]] = [(2, 1, 1)]  # root: (2x+1)(2y+1) = N
    forms_tested = 1

    if trace:
        print(f"Factoring N = {N}")
        print(f"  round  0: {len(live):>7} live form(s)   (place value a = 2)")
    if show_forms:
        print("  round  0  (place value a = 2):")
        print(f"      {format_form_line(2, 1, 1, N)}")

    round_num = 0
    while live:
        round_num += 1
        a_next = 2 ** (round_num + 1)  # place value of this round's children
        seen: set[tuple[int, int, int]] = set()
        children: list[tuple[int, int, int]] = []

        for a, r, s in live:
            for na, rp, sp in next_bit_forms(a, r, s):
                # Past sqrt(N) the smaller factor is below the place value, so it
                # must equal one of the residues; test them and drop the rest.
                if na > root:
                    for residue in (rp, sp):
                        if 1 < residue < N and N % residue == 0:
                            return found(residue)
                    continue

                survives, factor_found = analyze_form(na, rp, sp, N)
                if not survives:
                    continue
                # Collapse a form and its (s, r) swap into one.
                key = (na, min(rp, sp), max(rp, sp))
                if key in seen:
                    continue
                seen.add(key)
                children.append((na, rp, sp))

                forms_tested += 1
                if factor_found is not None:
                    return found(factor_found)

        if trace:
            print(
                f"  round {round_num:>2}: {len(children):>7} live form(s)   (place value a = {a_next})"
            )
        if show_forms:
            print(f"  round {round_num:>2}  (place value a = {a_next}):")
            for na, rp, sp in children:
                print(f"      {format_form_line(na, rp, sp, N)}")

        # End-of-round divisor probe: a factor equals one of the residues plus a
        # multiple of the place value; try the four nearest multiples.
        multiples = (a_next, 2 * a_next, 4 * a_next, 8 * a_next)
        for _, rp, sp in children:
            for residue in (rp, sp):
                for candidate in (residue + m for m in multiples):
                    if 1 < candidate < N and N % candidate == 0:
                        return found(candidate)

        live = children
        if round_num + 1 > N.bit_length():
            break  # exhausted every bit position without a factor

    if trace:
        print(f"  no factorization found after testing {forms_tested} forms")
    return None


def main() -> None:
    # Worked examples, each verified and timed with time.perf_counter().
    print("=== AERISBQ factoring ===\n")
    for N in (26329, 124169, 20082467, 100003 * 100019):
        start_time = time.perf_counter()
        result = factor(N)
        elapsed_time = time.perf_counter() - start_time
        assert result is not None and result[0] * result[1] == N
        p, q = result
        print(f"{N} = {p} x {q}   (run time: {elapsed_time:.3f} sec)")

    # Every surviving BQF, round by round: the Step-4 form it reduces to, a
    # solution, and whether that solution is positive (N = 26329 = 113 x 233):
    print("\n=== Surviving BQFs (with Step-4 form and solution) for N = 26329 ===")
    start_time = time.perf_counter()
    factor(26329, show_forms=True)
    elapsed_time = time.perf_counter() - start_time
    print(f"  (run time: {elapsed_time:.3f} sec)")

    # A compact per-round count -- the search grows, then collapses to a factor:
    print("\n=== Live-form count per round for N = 20082467 ===")
    start_time = time.perf_counter()
    factor(20082467, trace=True)
    elapsed_time = time.perf_counter() - start_time
    print(f"  (run time: {elapsed_time:.3f} sec)")

    # A range of semiprimes, balanced and lopsided alike:
    print("\n=== A range of semiprimes ===")
    for N in (4087, 143, 3599, 10403, 799819, 6422309):
        start_time = time.perf_counter()
        result = factor(N)
        elapsed_time = time.perf_counter() - start_time
        assert result is not None and result[0] * result[1] == N
        print(f"  {N} = {result[0]} x {result[1]}   (run time: {elapsed_time:.3f} sec)")

    # A balanced 56-bit semiprime -- two 28-bit primes of equal bit length. This
    # is AERISBQ's worst case: its cost is set by the smaller factor, and a
    # balanced N makes that factor as large as it can be (~sqrt(N)). The search
    # builds a frontier of roughly 2^(smaller factor's bit length) forms before it
    # lands a divisor -- here peaking near 16 million forms, about two minutes and
    # 5.5 GB. (Fermat's method has the opposite profile: it would crack a balanced
    # N quickly and struggle on a lopsided one. The two primes are kept apart, not
    # near-twin, so Fermat gets no easy foothold either.)
    print("\n=== A balanced 56-bit semiprime (deliberately slow: ~2 min, ~5.5 GB) ===")
    N = 52000010780000363  # 200000033 x 260000011
    start_time = time.perf_counter()
    result = factor(N)
    elapsed_time = time.perf_counter() - start_time
    assert result is not None and result[0] * result[1] == N
    print(f"  {N} ({N.bit_length()}-bit) = {result[0]} x {result[1]}")
    print(f"  (run time: {elapsed_time:.3f} sec)")


if __name__ == "__main__":
    main()
