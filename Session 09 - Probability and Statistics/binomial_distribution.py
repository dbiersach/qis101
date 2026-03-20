#!/usr/bin/env -S uv run
"""binomial_distribution.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import binom

# Number of trials
n = 50

# Array of integers from 0 to n (inclusive)
r_values = np.arange(n + 1)

# Use scipy.binom.pmf for each r_value for
# the given probabilities p=.3, p=.5, p=.9
dist1 = binom.pmf(r_values, n, 0.30)
dist2 = binom.pmf(r_values, n, 0.50)
dist3 = binom.pmf(r_values, n, 0.90)

# Use matplotlib bar plot to show each r_value
# with its associated probability of occurrence
plt.figure(Path(__file__).name)
plt.bar(r_values, dist1, label="p = 0.3")
plt.bar(r_values, dist2, label="p = 0.5")
plt.bar(r_values, dist3, label="p = 0.9")
plt.title(f"Binomial Distribution (n={n} trials)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
ax = plt.gca()
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(MultipleLocator(5))
plt.show()
