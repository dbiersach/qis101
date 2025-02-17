#!/usr/bin/env python3
"""circle_area.py"""

import numpy as np
import scipy.integrate


def f(x):
    return 4 * np.sqrt(1 - x**2)


def F(x):
    return 2 * (x * np.sqrt(1 - x**2) + np.arcsin(x))


def left_hand_rule(func, a, b, intervals):
    dx, area = (b - a) / intervals, 0.0
    for i in range(0, intervals):
        area += func(a + i * dx)
    return dx * area


def simpsons_rule(func, a, b, intervals):
    dx, area = (b - a) / intervals, func(a) + func(b)
    for i in range(1, intervals):
        area += func(a + i * dx) * (2 * (i % 2 + 1))
    return dx / 3 * area


def main():
    a, b, intervals = 0.0, 1.0, int(1e6)

    print("\nIntegrating 4 * sqrt(1 - x^2)", end=" ")
    print(f"over [{a}, {b}] using {intervals:,} intervals")

    area_analytic = F(b) - F(a)
    print(f"Exact Analytic : {area_analytic:.14f}\n")

    area_lh = left_hand_rule(f, a, b, intervals)
    print(f"Left-hand Rule : {area_lh:.14f}")
    print(
        f"Absolute % Relative Error : {abs(area_lh - area_analytic) / abs(area_analytic):.14%}\n"
    )

    area_simp = simpsons_rule(f, a, b, intervals)
    print(f"Simpson's Rule : {area_simp:.14f}")
    print(
        f"Absolute % Relative Error : {abs(area_simp - area_analytic) / abs(area_analytic):.14%}\n"
    )

    area_scipy = scipy.integrate.quad(f, a, b)[0]
    print(f"SciPy's quad() : {area_scipy:.14f}")
    print(
        f"Absolute % Relative Error : {abs(area_scipy - area_analytic) / abs(area_analytic):.14%}\n"
    )


if __name__ == "__main__":
    main()
