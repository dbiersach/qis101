#!/usr/bin/env python3
"""riemann_pi.py"""

from sympy import primepi

for n in range(1, 10):
    # Use sympy to calculate the Riemann Pi for 10^n
    riemann_pi = int(primepi(10**n))
    print(f"pi(10^{n}) = {riemann_pi:,}")
