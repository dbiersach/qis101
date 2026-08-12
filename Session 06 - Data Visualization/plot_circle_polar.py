#!/usr/bin/env -S uv run
"""plot_circle.polar.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Circle parameter
radius = 250

# Sample points around the circle
n_points = 10_000  # increase for better estimates
theta = np.linspace(0, 2 * np.pi, n_points)
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Area (shoelace formula)
shoelace_sum = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
area_est = abs(shoelace_sum) / 2

# Perimeter (accumulate Cartesian distances)
dx = np.diff(x)
dy = np.diff(y)
perimeter_est = np.sum(np.hypot(dx, dy))

print(f"Circle parameter: r={radius}")
print(f"Number of points: {n_points:,}")
print()
print(f"Estimated area: {area_est:,.8f}")
print(f"Pi from estimated area: {area_est / radius**2:,.8f}")
print()
print(f"Estimated perimeter: {perimeter_est:,.8f}")
print(f"Pi from estimated perimeter: {perimeter_est / (2 * radius):,.8f}")

plt.figure(Path(__file__).name)
plt.plot(x, y)
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.title(rf"Circle: $x^2 + y^2 = {radius}^2$")
plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.gca().set_aspect("equal")
plt.grid(True)
plt.show()
