#!/usr/bin/env -S uv run
"""complex_factorization.py"""

import numpy as np
from sympy import prime

num_odd_primes = 25

# Use a list comprehension to generate the first 'n' odd primes
# Note: in sympy, prime(1) = 2, prime(2) = 3, etc.
primes = [int(prime(n)) for n in range(2, num_odd_primes + 2)]

for p in primes:
    # Check if the prime 'p' can be expressed as a sum of two squares
    # As no prime is a perfect square, 'a' can be less than sqrt(p)
    # The +1 is because the range is exclusive of the upper bound
    for a in range(1, int(np.sqrt(p)) + 1):
        b = np.sqrt(p - a**2)
        # If 'b' is an integer, then 'p' is the sum of two squares
        if b == np.floor(b):
            z1 = complex(a, b)
            z2 = complex(a, -b)
            print(f"{p:>3} = {z1}{z2}")
            break
