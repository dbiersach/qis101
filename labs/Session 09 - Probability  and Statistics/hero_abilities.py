#!/usr/bin/env python3
"""hero_abilities.py"""

import numpy as np

n = 1_000_000

a = np.random.randint(1, 7, n)
a = a + np.random.randint(1, 7, n)
a = a + np.random.randint(1, 7, n)

print(f"{'Mean of 3d6':<16}:{np.mean(a):>6.2f}")
print(f"{'Std Dev of 3d6':<16}:{np.std(a):>6.2f}")

b = np.random.randint(3, 19, n)

print(f"{'Mean of 1d20':<16}:{np.mean(b):>6.2f}")
print(f"{'Std Dev of 1d20':<16}:{np.std(b):>6.2f}")
