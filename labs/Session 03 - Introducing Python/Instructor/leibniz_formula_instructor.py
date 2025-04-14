#!/usr/bin/env python3
"""leibniz_formula_instructor.py"""

import numpy as np

n = np.arange(1_000_000)

x = (-1) ** n
y = 2 * n + 1

print(4 * np.sum(x / y))
