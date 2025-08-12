#!/usr/bin/env python3
'''big_sqrt.py'''

from mpmath import fabs, mp, mpf


def sqrt(x):
    low, high = mpf("0.0"), x
    est = (high + low) / 2
    epsilon = mpf(10 ** (-mp.dps / 2))
    while fabs(est**2 - x) > epsilon:
        if est**2 > x:
            high = est
        else:
            low = est
        est = (high + low) / 2
    return est


def main():
    mp.dps = 200  # dps = decimal places
    x = mpf(
        (
            "33590351381261822622218163873528556813698947665687"
            "61568876758902106044097938012929232223664368425159"
        )
    )
    print(f"x = {x}")
    print(f"sqrt(x) = {sqrt(x)}")


if __name__ == "__main__":
    main()
