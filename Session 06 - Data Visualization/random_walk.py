#!/usr/bin/env -S uv run
"""random_walk.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2017)
n = 10_000
angles = 2 * np.pi * np.random.rand(n)
x = np.cumsum(np.cos(angles))
y = np.cumsum(np.sin(angles))

plt.figure(Path(__file__).name)
plt.plot(x, y)
plt.plot(x[0], y[0], color="green", marker="o")
plt.plot(x[-1], y[-1], color="red", marker="o")
# fmt: off
plt.arrow(x[0], y[0], x[-1] - x[0], y[-1] - y[0],
        color="black", linestyle="--",  width=0.3,
        head_width=1,  length_includes_head=True, zorder=3)
# fmt: on
plt.title(f"Uniform Random Walk ({n:,} Unit Steps)")
plt.gca().set_aspect("equal")
plt.show()
