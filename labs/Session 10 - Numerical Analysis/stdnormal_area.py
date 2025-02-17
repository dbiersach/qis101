#!/usr/bin/env python3
"""stdnormal_area.py"""

import numpy as np
import scipy.integrate


def f(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)


def main():
    area = scipy.integrate.quad(f, -1, 1)[0]

    # See https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    print(f"\nNormal CDF: Probability X is within ± 1st sigma = {area:.8%}\n")


if __name__ == "__main__":
    main()
