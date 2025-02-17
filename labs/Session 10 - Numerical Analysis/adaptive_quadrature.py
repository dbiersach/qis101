#!/usr/bin/env python3
"""adaptive_quadrature.py"""

import time


def f(x):
    return 5 * pow(x, 3) - 9 * pow(x, 2) + 11


def F(x):
    return 5 * pow(x, 4) / 4 - 3 * pow(x, 3) + 11 * x


def midpoint_fixed(a, b):
    x, dx, area = a, (b - a) / 1e6, 0.0
    while x < b:
        area += f(x + dx / 2) * dx
        x += dx
    return area


def midpoint_adaptive(a, b):
    x, dx, area = a, 1, 0.0
    while x < b:
        f1 = f(x)
        f2 = f(x + dx)
        # Keep halving dx if current delta is too great
        while abs(((f2 - f1) / f1)) > 1e-3:
            dx /= 2
            f2 = f(x + dx)
        # Use the midpoint rule
        area += f(x + dx / 2) * dx
        x += dx
        # Expand current interval width
        dx *= 2
    return area


def main():
    a, b = 0.0, 10.0

    area_analytic = F(b) - F(a)
    print(f"\nExact integral using analytic calculus = {area_analytic}\n")

    start_time = time.perf_counter()
    area_fixed = midpoint_fixed(a, b)
    elapsed_time = time.perf_counter() - start_time
    err = abs(area_fixed - area_analytic) / abs(area_analytic)
    print(f"Estimated integral using fixed width midpoint rule = {area_fixed}")
    print(f"Fixed width midpoint rule % error = {err:.8%}")
    print(f"Time (sec) Fixed = {elapsed_time:.3f}\n")

    start_time = time.perf_counter()
    area_adaptive = midpoint_adaptive(a, b)
    elapsed_time = time.perf_counter() - start_time
    err = abs(area_adaptive - area_analytic) / abs(area_analytic)
    print(f"Estimated integral using adaptive width midpoint rule = {area_adaptive}")
    print(f"Adaptive width midpoint rule % error = {err:.8%}")
    print(f"Time (sec) Adaptive = {elapsed_time:.3f}\n")


if __name__ == "__main__":
    main()
