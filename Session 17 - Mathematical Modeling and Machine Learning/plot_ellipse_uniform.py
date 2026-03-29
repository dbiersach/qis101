#!/usr/bin/env -S uv run
"""plot_ellipse_uniform.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

a, b = 250, 150
num_points = 50

# Step 1: Compute arc length as a function of theta using many sample points
theta_fine = np.linspace(0, 2 * np.pi, 10000)

# The differential arc length ds/dθ for an ellipse is sqrt((a*sin(θ))² + (b*cos(θ))²)
ds_dtheta = np.sqrt((a * np.sin(theta_fine)) ** 2 + (b * np.cos(theta_fine)) ** 2)

# Step 2: Cumulative arc length
s = np.concatenate([[0], cumulative_trapezoid(ds_dtheta, theta_fine)])

# Step 3: Interpolate to find theta values at equally spaced arc lengths
s_uniform = np.linspace(0, s[-1], num_points, endpoint=False)
theta_uniform = np.interp(s_uniform, s, theta_fine)

# Step 4: Compute (x, y) at those theta values
x = a * np.cos(theta_uniform)
y = b * np.sin(theta_uniform)

# Close the ellipse by appending the first point at the end
x = np.append(x, x[0])
y = np.append(y, y[0])

# Plot
plt.figure(Path(__file__).name)
plt.plot(x, y, "o-")  # 'o-' to see the individual points
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
