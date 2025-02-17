#!/usr/bin/env python3
"""simpsons_rule.py"""


def f(x):
    return (x + 9) * (x + 4) * (x + 1) * (x - 5) * (x - 11)


def F(x):
    # fmt: off
    return (1 / 6 * pow(x, 6) - 2 / 5 * pow(x, 5) - 30 * pow(x, 4)
                + 22 / 3 * pow(x, 3) + 2119 / 2 * pow(x, 2) + 1980 * x)
    # fmt: on


def left_hand_rule(func, a, b, intervals):
    dx = (b - a) / intervals
    area = 0.0
    for i in range(0, intervals):
        area += func(a + i * dx)
    return dx * area


def simpsons_rule(func, a, b, intervals):
    dx, area = (b - a) / intervals, func(a) + func(b)
    for i in range(1, intervals):
        area += func(a + i * dx) * (2 * (i % 2 + 1))
    return dx / 3 * area


def main():
    a, b = -10.0, 12.0
    intervals = int(1e6)

    print("\nIntegrating x^5 - 2x^4 - 120x^3 + 22x^2 + 2199x + 1980")
    print(f" over [{a}, {b}] using {intervals:,} intervals\n")

    area_act = F(b) - F(a)
    print(f"Analytic (Exact) : {area_act:.14f}\n")

    area_lh = left_hand_rule(f, a, b, intervals)
    print(f"Left-hand Rule   : {area_lh:.14f}")
    print(
        f"Absolute % Relative Error : {abs(area_lh - area_act) / abs(area_act):.14%}\n"
    )

    area_simpsons = simpsons_rule(f, a, b, intervals)
    print(f"Simpson's Rule   : {area_simpsons:.14f}")
    print(
        f"Absolute % Relative Error : {abs(area_simpsons - area_act) / abs(area_act):.14%}\n"
    )


if __name__ == "__main__":
    main()
