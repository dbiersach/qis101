#!/usr/bin/env python3
"""euler_totient.py"""

import numpy as np


def totient(n):
    s = 1
    for i in range(2, n):
        if np.gcd(i, n) == 1:
            s += 1
    return s


def main():
    print("Natural numbers between 2 and 100", end=" ")
    print("that exceed their totient by one:")

    for n in range(2, 101):
        if n == totient(n) + 1:
            print(n, end=" ")


if __name__ == "__main__":
    main()
