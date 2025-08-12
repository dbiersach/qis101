#!/usr/bin/env python3
"""make_prime_powers.py"""

import pickle
from pathlib import Path
from pprint import pprint

from sympy import primerange

# Create a dictionary where the keys are all the primes < n and each
# key's value is a *set* of all the powers of that prime that are also < n
n = 10_000
primes: list[int] = [int(p) for p in primerange(2, n)]
powers_dict: dict[int, list[int]] = {}
for p in primes:
    powers_set: set[int] = set()
    power = p
    while power < n:
        powers_set.add(power)
        power *= p
    powers_dict[p] = sorted(powers_set)

# Pretty-print the fist 10 key/value pairs in the prime powers dictionary
pprint(dict(list(powers_dict.items())[:10]))

# Write pickle file of the prime powers dictionary
file_path = Path(__file__).parent / "prime_powers.pickle"
with open(file_path, "wb") as file_out:
    pickle.dump(powers_dict, file_out, pickle.HIGHEST_PROTOCOL)
print(f"Created {file_path}")
