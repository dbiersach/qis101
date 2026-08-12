#!/usr/bin/env -S uv run
"""plot_infinite_series.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

n = 100_000
x = np.linspace(1, n, n)
y1 = np.cumsum(1 / x)
y2 = np.cumsum(1 / x**2)
print(np.sqrt(6 * y2[-1]))

plt.figure(Path(__file__).name)
plt.plot(x, y1, label=r"$\frac{1}{x}$")
plt.plot(x, y2, label=r"$\frac{1}{x^2}$")
plt.title("Infinite Series")
plt.xlabel("Number of Terms")
plt.ylabel("Cumulative Sum")
plt.legend(loc="center right")
plt.show()
