#!/usr/bin/env -S uv run
"""bernoulli_series.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

terms = 14
x = np.arange(1, terms + 1)
y1 = np.zeros_like(x, dtype=np.float64)
y2 = np.zeros_like(x, dtype=np.float64)

# See https://en.wikipedia.org/wiki/Sophomore%27s_dream
for n in range(1, terms):
    y1[n] = y1[n - 1] + n ** (-n)
    dx = 1.0 / n
    x_mid = (np.arange(n) + 0.5) * dx
    y2[n] = dx * np.sum(x_mid ** (-x_mid))

plt.figure(Path(__file__).name)
plt.plot(x, y1, label=r"$\sum_{n=1}^{\infty} n^{-n}$")
plt.plot(x, y2, label=r"$\int_0^1 x^{-x}\,dx$")
plt.title("Johann Bernoulli's Sophomore's Dream Identity")
plt.xlabel("Number of terms / quadrature nodes")
plt.ylabel("Approximation")
plt.annotate(
    f"{y1[-1]:.14f}",  # label text: final value to 14 decimal places
    (x[-1], y1[-1]),  # anchor point in data coordinates (last x, last y)
    xytext=(-5, -15),  # label offset: 5 px left, 15 px down from anchor
    textcoords="offset points",  # interpret xytext as a pixel offset, not data coords
    ha="right",  # right-align text so it ends at the offset position
    color="C0",  # match the first line in matplotlib's default color cycle
)
plt.annotate(
    f"{y2[-1]:.14f}",
    xy=(x[-1], y2[-1]),
    xytext=(-5, 10),
    textcoords="offset points",
    ha="right",
    color="C1",
)
plt.legend(loc="center right")
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid("on")
plt.show()
