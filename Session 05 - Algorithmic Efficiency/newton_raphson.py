#!/usr/bin/env -S uv run
"""newton_raphson.py"""

import timeit

import numpy as np

iterations = 30


def bisection(x):
    low, high = 0, x
    est = (low + high) / 2
    for _ in range(iterations):
        if est**2 < x:
            low = est
        else:
            high = est
        est = (low + high) / 2
    return est


def newton_raphson(f, df, x0):
    est = x0
    for _ in range(iterations):
        est = est - f(est) / df(est)
    return est


def main():
    x = 1234567890123456.78

    r0 = np.sqrt(x)
    r1 = bisection(x)
    r2 = newton_raphson(lambda v: v**2 - x, lambda v: 2 * v, x / 2)

    print(f"        Actual sqrt(X) = {r0:.12f}")
    print(f"     Bisection sqrt(x) = {r1:.12f}")
    print(f"Newton-Raphson sqrt(x) = {r2:.12f}")

    # Time each method
    trials = 100
    t_bisection = timeit.timeit(lambda: bisection(x), number=trials)
    t_newton = timeit.timeit(
        lambda: newton_raphson(lambda v: v**2 - x, lambda v: 2 * v, x / 2),
        number=trials,
    )
    print(f"     Bisection: {t_bisection:.4f} seconds")
    print(f"Newton-Raphson: {t_newton:.4f} seconds")


if __name__ == "__main__":
    main()
