#!/usr/bin/env -S uv run
"""plot_ellipse_polar.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ellipse parameters
a, b = 250, 150

# Sample points around the ellipse
n_points = 10_000  # increase for better estimates
theta = np.linspace(0, 2 * np.pi, n_points)

radius = a * b / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Area (shoelace formula)
shoelace_sum = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
area_est = abs(shoelace_sum) / 2

# Perimeter (accumulate Cartesian distances)
dx = np.diff(x)
dy = np.diff(y)
perimeter_est = np.sum(np.hypot(dx, dy))

print(f"Ellipse parameters: a={a}, b={b}")
print(f"Number of points: {n_points:,}")
print()
print(f"Estimated area: {area_est:,.8f}")
print(f"Pi from estimated area: {area_est / (a * b):,.8f}")
print()
print(f"Estimated perimeter: {perimeter_est:,.8f} (Cartesian distance sum)")
print(
    "Estimated perimeter: "
    f"{np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))):,.8f}"
    " (Ramanujan's formula)"
)

# Plot
plt.figure(Path(__file__).name)
plt.plot(x, y)
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
