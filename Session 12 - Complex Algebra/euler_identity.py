#!/usr/bin/env -S uv run
"""euler_identity.py"""

import numpy as np
import scipy

x = np.arange(20)

n = np.power(complex(0, np.pi), x)
d = scipy.special.factorial(x)
ez = np.sum(n / d)

ez = np.round(ez, 8)
print(ez)
