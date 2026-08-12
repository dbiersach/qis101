#!/usr/bin/env -S uv run
"""plot_ellipse.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ellipse parameters
a, b = 250, 150

# Domain is 1st quadrant: x >= 0, y >= 0
x = np.linspace(0, a, 1000)
# Standard ellipse equation rearranged to solve for y
y = b * np.sqrt(1 - (x**2 / a**2))

plt.figure(Path(__file__).name)
plt.plot(x, y)
# Use symmetry to plot the other three quadrants
plt.plot(x, -y)
plt.plot(-x, y)
plt.plot(-x, -y)
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.title(rf"Ellipse: $\dfrac{{x^2}}{{{a}^2}}+\dfrac{{y^2}}{{{b}^2}}=1$")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-a - 50, a + 50)
plt.ylim(-b - 50, b + 50)
plt.gca().set_aspect("equal")
plt.grid(True)
plt.show()
