#!/usr/bin/env -S uv run
"""plot_circle.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

radius = 250
# Domain is 1st quadrant: x >= 0, y >= 0
x = np.linspace(0, radius, 1000)
# Standard circle equation rearranged to solve for y
y = np.sqrt(radius**2 - x**2)

plt.figure(Path(__file__).name)
plt.plot(x, y)
# Use symmetry to plot the other three quadrants
plt.plot(x, -y)
plt.plot(-x, y)
plt.plot(-x, -y)
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.title(rf"Circle: $x^2 + y^2 = {radius}^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.gca().set_aspect("equal")
plt.grid(True)
plt.show()
