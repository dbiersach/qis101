#!/usr/bin/env -S uv run
"""euler_identity.py"""

import numpy as np
from scipy.special import factorial

x = np.arange(20)

n = np.power(complex(0, np.pi), x)
d = factorial(x)
ez = np.sum(n / d)

ez = np.round(ez, 8)
print(ez)
