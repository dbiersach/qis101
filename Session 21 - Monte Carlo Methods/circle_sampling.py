#!/usr/bin/env -S uv run
"""circle_sampling.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

n = 10_000
t = np.random.rand(n) * 2 * np.pi
r = np.random.rand(n)
x = r * np.cos(t)
y = r * np.sin(t)

plt.figure(Path(__file__).name)
plt.scatter(x, y, s=0.5)
plt.title("Uniform Circle Sampling")
plt.gca().set_aspect("equal")
plt.show()
