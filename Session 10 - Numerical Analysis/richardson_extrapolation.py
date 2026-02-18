#!/usr/bin/env -S uv run
"""richardson_extrapolation.py"""

import numpy as np


def f(x):
    return np.sin(x)


def df(x):
    return np.cos(x)


def D2(f, x, h):
    """2nd Order Centered Differences"""
    return (f(x + h) - f(x - h)) / (2 * h)


def D4(f, x, h):
    """4th Order Centered Differences"""
    t1 = -f(x + 2 * h)
    t2 = 8 * f(x + h)
    t3 = -8 * f(x - h)
    t4 = f(x - 2 * h)
    return (t1 + t2 + t3 + t4) / (12 * h)


def D_Rich(f, x, D, h, p):
    """Richardson Extrapolation"""
    t1 = D(f, x, h)
    t2 = D(f, x, h / 2)
    return (2**p * t2 - t1) / (2**p - 1)


def main():
    x0 = 1.0
    h = 0.01

    exact = df(x0)

    est_2nd_basic = D2(f, x0, h)
    est_4th_basic = D4(f, x0, h)

    est_2nd_rich = D_Rich(f, x0, D2, h, 2)
    est_4th_rich = D_Rich(f, x0, D4, h, 4)

    print(f"\n{'Exact derivative':<27}: {exact:.16f}\n")

    print(f"{'Basic 2nd order CD':<27}: {est_2nd_basic:.16f}")
    print(f"{'APE (Basic 2nd Order)':<27}", end=": ")
    print(f"{abs((est_2nd_basic - exact) / exact) * 100:.16%}\n")

    print(f"{'Basic 4th order CD':<27}: {est_4th_basic:.16f}")
    print(f"{'APE (Basic 4th Order)':<27}", end=": ")
    print(f"{abs((est_4th_basic - exact) / exact) * 100:.16%}\n")

    print(f"{'Richardson 2nd order CD':<27}: {est_2nd_rich:.16f}")
    print(f"{'APE (Richardson 2nd Order)':<27}", end=": ")
    print(f"{abs((est_2nd_rich - exact) / exact) * 100:.16%}\n")

    print(f"{'Richardson 4th order CD':<27}: {est_4th_rich:.16f}")
    print(f"{'APE (Richardson 4th Order)':<27}", end=": ")
    print(f"{abs((est_4th_rich - exact) / exact) * 100:.16%}\n")


if __name__ == "__main__":
    main()
