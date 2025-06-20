#!/usr/bin/env python3
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

    print("Exact derivative             : ", exact)
    print()

    print("Basic 2nd order CD           : ", est_2nd_basic)
    print("Error (Basic 2nd Order)      : ", abs(est_2nd_basic - exact))
    print()

    print("Basic 4th order CD           : ", est_4th_basic)
    print("Error (Basic 4th Order)      : ", abs(est_4th_basic - exact))
    print()

    print("Richardson 2nd order CD      : ", est_2nd_rich)
    print("Error (Richardson 2nd Order) : ", abs(est_2nd_rich - exact))
    print()

    print("Richardson 4th order CD      : ", est_4th_rich)
    print("Error (Richardson 4th Order) : ", abs(est_4th_rich - exact))
    print()


if __name__ == "__main__":
    main()
