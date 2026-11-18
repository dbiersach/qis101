#!/usr/bin/env -S uv run
"""big_sqrt.py"""

from mpmath import mp, mpf


def sqrt(x):
    if not mp.isfinite(x) or x < 0:
        raise ValueError("Expected a finite nonnegative number")
    if x == 0:
        return mpf(0)
    low, high = mpf(0), max(mpf(1), x)
    est = (high + low) / 2
    epsilon = mpf(10) ** (-(mp.dps // 2))
    while abs(est - x / est) > epsilon * est:
        if est**2 > x:
            high = est
        else:
            low = est
        next_est = (high + low) / 2
        if next_est == est:
            break
        est = next_est
    return est


def main():
    mp.dps = 200  # dps = decimal places
    x = mpf(
        "33590351381261822622218163873528556813698947665687"
        "61568876758902106044097938012929232223664368425159"
    )
    print(f"x = {x}")
    print(f"sqrt(x) = {sqrt(x)}")


if __name__ == "__main__":
    main()
