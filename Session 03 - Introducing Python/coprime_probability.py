#!/usr/bin/env -S uv run
"""coprime_probability.py"""

import numpy as np

n = 1_000_000
a = np.random.randint(1, n, size=n)
b = np.random.randint(1, n, size=n)

c = np.gcd(a, b)
p = np.sum(c == 1) / n

print(f"Coprime Probability = {p:.2%}")
print(f"Hidden constant     = {np.sqrt(6 / p):.3f}")
