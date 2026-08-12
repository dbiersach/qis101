#!/usr/bin/env -S uv run
"""gauss_summation.py"""

import numpy as np

n = 10
x = np.arange(1, n + 1)
print(f"x = {x}")

y1 = np.cumsum(x)
print(f"y1 = {y1}")

y2 = x * (x + 1) / 2
print(f"y2 = {y2}")

print(f"y1 == y2 ? {np.array_equal(y1, y2)}")
