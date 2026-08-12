#!/usr/bin/env -S uv run
"""dirichlet_function.py"""

import mpmath


def dirichlet_function(x):
    k, n = 100.0, 1e10
    val = mpmath.cos(mpmath.factorial(k) * mpmath.pi * x)
    val = mpmath.power(val, 2 * n)
    val = mpmath.chop(val)
    return val


def main():
    mpmath.mp.dps = 2000  # dps = decimal places
    print(f"f(2) = {dirichlet_function(2)}")
    print(f"f(2.5) = {dirichlet_function(2.5)}")
    print(f"f(sqrt(2)) = {dirichlet_function(mpmath.sqrt(2))}")
    print(f"f(e) = {dirichlet_function(mpmath.e)}")


if __name__ == "__main__":
    main()
