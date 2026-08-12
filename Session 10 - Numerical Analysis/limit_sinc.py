#!/usr/bin/env -S uv run
"""limit_sinc.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Calculate sinc function values
x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
y = np.sin(x) / x

# Plot the sinc function
plt.figure(Path(__file__).name, figsize=(10, 6))

plt.plot(x, y, "b-", linewidth=1.5, label=r"$\mathrm{sinc}(x) = \dfrac{\sin(x)}{x}$")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

# Mark the limit point
plt.plot(
    0, np.max(y), "ro", markersize=8, label=r"$\lim_{x \to 0} \dfrac{\sin(x)}{x} = 1$"
)

# Find local minima (where derivative changes from negative to positive)
dy = np.diff(y)
minima_idx = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1
minima_values = y[minima_idx]

# Get the two most negative minima
two_lowest_idx = minima_idx[np.argsort(minima_values)[:2]]
min_x = x[two_lowest_idx[0]]
min_y = y[two_lowest_idx[0]]
plt.plot(
    x[two_lowest_idx],
    y[two_lowest_idx],
    "go",
    markersize=8,
    label=rf"Global minima at $(\pm{min_x:.2f}, {min_y:.4f})$",
)

# Find first zero crossings (where y changes sign, closest to x=0)
zero_crossings = np.where(np.diff(np.sign(y)))[0]
# Get the two closest to x=0 (one negative, one positive)
crossing_x = x[zero_crossings]
first_pos_crossing_idx = zero_crossings[crossing_x > 0][0]
first_neg_crossing_idx = zero_crossings[crossing_x < 0][-1]
plt.plot(
    [x[first_neg_crossing_idx], x[first_pos_crossing_idx]],
    [0, 0],
    "ko",
    markersize=8,
    label=r"First zero crossings at $x = \pm\pi$",
)

# Set title, labels, and limits
plt.title(
    r"The Sinc Function: $\mathrm{sinc}(x) = \dfrac{\sin(x)}{x}$",
    fontsize=14,
)
plt.xlabel(r"$x$ (radians)", fontsize=12)
plt.ylabel(r"$y$", fontsize=12)
plt.xlim(-4 * np.pi, 4 * np.pi)
plt.ylim(-0.30, 1.10)

# Set major ticks with 2 decimal places
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(np.pi))
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f"{val:.2f}" if val != 0 else "0.00")
)
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.25))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))

# Set minor ticks
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))

# Enable grid
plt.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.9)
plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.8)

# Add legend
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()
