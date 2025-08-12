#!/usr/bin/env python3
'''logarithmic_series.py'''

import numpy as np

k = np.arange(1, 31)
s = np.sum(1 / (2**k * k))
print(np.exp(s))
