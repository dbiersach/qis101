#!/usr/bin/env python3
"""gamma_eta.py"""

import numpy as np
from mpmath import altzeta
from scipy.integrate import quad
from scipy.special import factorial

# We cannot raise e^x to a greater exponent value than this
max_exponent = int(np.log(np.finfo(np.longdouble).max) - 1)


def f(x, s):
    if x < max_exponent:
        return np.power(x, s - 1) / (np.exp(x) + 1)
    return 0.0


def main():
    # We want to calculate 5!
    n = 5

    # Gamma[n+1] == n!
    s = n + 1

    # Calculate the Dirichlet's Eta value for s
    # See https://en.wikipedia.org/wiki/Dirichlet_eta_function
    eta = float(altzeta(s))

    # Use scipy to integrate f(x, s) from 0 to 1000
    integral = quad(f, 0, 1000, args=(s,))[0]

    # Calculate gamma, given both the eta and integral values
    # See https://en.wikipedia.org/wiki/Gamma_function
    gamma = int(integral / eta)

    print(f"{n}! = {gamma}")
    print(f"{n}! = {factorial(n, exact=True)}")


if __name__ == "__main__":
    main()
