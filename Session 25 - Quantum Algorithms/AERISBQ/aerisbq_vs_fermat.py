#!/usr/bin/env -S uv run
"""aerisbq_vs_fermat.py"""

# Compare AERISBQ against Fermat's difference-of-squares method on the semiprimes
# in semiprimes.csv (run create_semiprimes.py first). The two methods are hard on
# opposite inputs: AERISBQ's cost grows with the SMALLER prime, so it struggles on
# balanced N; Fermat's cost grows with (sqrt(q) - sqrt(p))^2, so it struggles on
# lopsided N.
#
# Raw elapsed time is an unfair yardstick because a Fermat step (one squaring plus
# a perfect-square test) is far cheaper than an AERISBQ step (a full four-step
# sieve). So we track two things per factorization:
#
#   * elapsed time      -- the wall clock, the "simpleton" metric.
#   * work effort       -- the number of candidate hypotheses each method examines:
#                          AERISBQ counts the binary quadratic forms it carries,
#                          Fermat counts the values of a it tries. This abstracts
#                          away the per-step cost and measures algorithmic effort.
#
# Both methods run in numba and stop once they reach WORK_BUDGET hypotheses, so the
# infeasible tails (Fermat on a very lopsided N, AERISBQ on a large balanced N) are
# bounded rather than left to run forever.

import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aerisbq_factoring_numba import _analyze, _isqrt
from numba import njit

WORK_BUDGET = 1_000_000  # stop each method after this many candidate hypotheses
PACK = np.int64(1) << 31  # pack a form's (r, s) into one int64 dedup key
CROSSOVER_BITS = 50  # bit length shown in the per-input crossover panel
CROSSOVER_BUDGET = 50_000_000  # crossover panel runs AERISBQ with no effective cap


# The @njit functions below are left unannotated: numba types them from the int64
# values it is handed, and Python `int` hints clash with those numpy scalars under
# a static type checker. The pure-Python helpers keep their hints.


@njit(cache=True)
def fermat_work(N, max_iters):
    """Fermat's method. Returns (factor, iterations); factor 0 means budget hit."""
    a = _isqrt(N)
    if a * a < N:
        a += 1
    for it in range(1, max_iters + 1):
        b2 = a * a - N
        b = _isqrt(b2)
        if b * b == b2:
            f = a - b
            if 1 < f < N:
                return f, it
        a += 1
    return 0, max_iters


@njit(cache=True)
def aerisbq_work(N, max_forms):
    """AERISBQ search. Returns (factor, forms carried); factor 0 means budget hit."""
    if N % 2 == 0:
        return 2, 1
    root = _isqrt(N)
    r_live = np.empty(1, np.int64)
    s_live = np.empty(1, np.int64)
    r_live[0] = 1
    s_live[0] = 1
    a = np.int64(2)
    forms = 1

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
                if na > root:
                    if 1 < rp < N and N % rp == 0:
                        return rp, forms
                    if 1 < sp < N and N % sp == 0:
                        return sp, forms
                    continue
                survives, fac = _analyze(na, rp, sp, N)
                if not survives:
                    continue
                key = min(rp, sp) * PACK + max(rp, sp)
                if key in seen:
                    continue
                seen.add(key)
                r_child[cnt] = rp
                s_child[cnt] = sp
                cnt += 1
                forms += 1
                if fac != 0:
                    return fac, forms
                if forms >= max_forms:
                    return 0, forms
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
                    return cand, forms
        if cnt == 0:
            break
        r_live = r_child[:cnt].copy()
        s_live = s_child[:cnt].copy()
        a = na

    return 0, forms


def fermat_iterations(N: int, p: int, q: int) -> int:
    """Exact Fermat iteration count for N = p*q -- a closed form, no simulation.

    Fermat scans a upward from ceil(sqrt(N)) and stops at the first a whose
    a*a - N is a perfect square. A semiprime has two factorizations: 1*N, which
    is rejected because it yields the factor 1, and p*q, which gives a = (p+q)/2.
    That is the first a Fermat accepts, so the number of steps to reach it -- the
    method's true work -- is a plain subtraction, exact for any N within int range.
    """
    a0 = math.isqrt(N)
    if a0 * a0 < N:
        a0 += 1
    return (p + q) // 2 - a0 + 1


def timed(func, N: int, budget: int) -> tuple[int, float]:
    """Run func(N, budget), returning (work_done, elapsed_seconds)."""
    start_time = time.perf_counter()
    _factor, work = func(N, budget)
    elapsed_time = time.perf_counter() - start_time
    return int(work), elapsed_time


def main() -> None:
    file_path = Path(__file__).parent / "semiprimes.csv"
    with open(file_path, newline="") as f:
        rows = [(int(N), int(p), int(q)) for N, p, q in list(csv.reader(f))[1:]]
    print(f"Read {len(rows)} semiprimes from {file_path}")

    # Warm up both JIT compilers so compilation is excluded from the timings.
    fermat_work(15, 10)
    aerisbq_work(15, 10)

    bit_lengths = sorted({N.bit_length() for N, _, _ in rows})
    n_bits = len(bit_lengths)
    # Per bit length: summed work and time for each method.
    aer_work = dict.fromkeys(bit_lengths, 0)
    aer_time = dict.fromkeys(bit_lengths, 0.0)
    fer_work = dict.fromkeys(bit_lengths, 0)
    fer_time = dict.fromkeys(bit_lengths, 0.0)
    # Detail for the crossover panel (one chosen N bit length): the work each
    # method spends, grouped by the smaller prime's bit length.
    cross_aer: dict[int, list[int]] = {}
    cross_fer: dict[int, list[int]] = {}

    for idx, (N, p, q) in enumerate(rows):
        b = N.bit_length()
        aw, at = timed(aerisbq_work, N, WORK_BUDGET)
        fw, ft = timed(fermat_work, N, WORK_BUDGET)
        aer_work[b] += aw
        aer_time[b] += at
        fer_work[b] += fw
        fer_time[b] += ft
        if b == CROSSOVER_BITS:
            # This panel shows each method's true effort with no budget cap:
            # AERISBQ runs to completion (its balanced cost tops WORK_BUDGET) and
            # Fermat uses its exact closed-form count, so neither curve flattens.
            aw_full, _ = timed(aerisbq_work, N, CROSSOVER_BUDGET)
            cross_aer.setdefault(p.bit_length(), []).append(aw_full)
            cross_fer.setdefault(p.bit_length(), []).append(fermat_iterations(N, p, q))
        if (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(rows)} semiprimes measured")

    print("Plotting...")
    xs = np.array(bit_lengths)
    fig, (ax_t, ax_w, ax_c) = plt.subplots(3, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title(Path(__file__).name)

    # Panel 1: total elapsed time per bit length (the "simpleton" metric).
    ax_t.semilogy(xs, [aer_time[b] for b in bit_lengths], "o-", label="AERISBQ")
    ax_t.semilogy(xs, [fer_time[b] for b in bit_lengths], "s-", label="Fermat")
    ax_t.set_title(
        f"Total elapsed time per bit length ({len(rows) // n_bits} semiprimes each)"
    )
    ax_t.set_xlabel("bit length of N")
    ax_t.set_ylabel("total time (sec)")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend()

    # Panel 2: total work effort per bit length (the fair abstract metric).
    ax_w.semilogy(xs, [aer_work[b] for b in bit_lengths], "o-", label="AERISBQ (forms)")
    ax_w.semilogy(
        xs, [fer_work[b] for b in bit_lengths], "s-", label="Fermat (iterations)"
    )
    ax_w.axhline(
        (len(rows) // n_bits) * WORK_BUDGET,
        color="gray",
        ls="--",
        alpha=0.6,
        label="budget ceiling",
    )
    ax_w.set_title("Total work effort per bit length (hypotheses examined)")
    ax_w.set_xlabel("bit length of N")
    ax_w.set_ylabel("total hypotheses")
    ax_w.grid(True, which="both", alpha=0.3)
    ax_w.legend()

    # Panel 3: the crossover -- at a fixed N bit length, work versus how balanced N
    # is. Several semiprimes can share one smaller-prime bit length, so their work
    # is averaged into a single point per bit length. That makes the panel a true
    # function of p's size and matches the per-bit-length aggregation above; plotting
    # the raw points instead would draw a vertical segment wherever two share an x.
    cx = sorted(cross_aer)
    aer_mean = [sum(cross_aer[k]) / len(cross_aer[k]) for k in cx]
    fer_mean = [sum(cross_fer[k]) / len(cross_fer[k]) for k in cx]
    ax_c.semilogy(cx, aer_mean, "o-", label="AERISBQ (forms)")
    ax_c.semilogy(cx, fer_mean, "s-", label="Fermat (exact iterations)")
    ax_c.set_title(
        f"Work vs balance at {CROSSOVER_BITS}-bit N (true effort, no budget cap)"
    )
    ax_c.set_xlabel("bit length of the smaller prime p  (lopsided -> balanced)")
    ax_c.set_ylabel("hypotheses examined")
    ax_c.grid(True, which="both", alpha=0.3)
    ax_c.legend()

    plt.tight_layout()
    out_path = Path(__file__).parent / "aerisbq_vs_fermat.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
