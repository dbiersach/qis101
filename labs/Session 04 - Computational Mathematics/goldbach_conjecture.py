#!/usr/bin/env python3
"""goldbach_conjecture.py"""

from itertools import combinations_with_replacement

import numpy as np
from sympy import prime

test_limit = 100

print("Verifying Goldbach's conjecture for every EVEN ", end="")
print(f"integer from 6 to {test_limit} inclusive:")

# Use a list comprehension to create an array of the first 'n' primes
primes = np.array([prime(n) for n in range(2, test_limit)])

# Generate all pairs of primes (with repetition)
prime_pairs = [*combinations_with_replacement(primes, 2)]

# Create sorted array containing the unique sums of each prime pairs
summed_pairs = np.sort(np.unique(np.array([sum(pair) for pair in prime_pairs])))

# Determine which EVEN integers from 6 to n (inclusive)
# are NOT in the list of summed prime pairs
# The numpy function setdiff1d() returns any elements in the first list
# that are not also in the second list
violations = np.setdiff1d(range(6, test_limit + 2, 2), summed_pairs)

if len(violations) == 0:
    print("No Goldbach violations were found")
else:
    print(f"Found these violations: {violations}")
