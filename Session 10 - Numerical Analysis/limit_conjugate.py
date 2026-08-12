#!/usr/bin/env -S uv run
"""limit_conjugate.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Calculate sqrt(x^2 + x) - x for positive x
x = np.linspace(0.0, 20, 1000)
y = np.sqrt(x**2 + x) - x

# Plot the function
plt.figure(Path(__file__).name, figsize=(10, 6))
plt.plot(x, y, "b-", linewidth=1.5, label=r"$f(x) = \sqrt{x^2 + x} - x$")

# Mark the limit value y = 1/2
plt.plot(
    x[-10],
    0.5,
    "ro",
    markersize=8,
    label=r"$\lim_{x \to \infty} \left(\sqrt{x^2 + x} - x\right) = \dfrac{1}{2}$",
)
# Horizontal asymptote at y = 1/2
plt.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)

# Set title, labels, and limits
plt.title(
    r"A Radical Difference Limit: $\lim_{x \to \infty} \left(\sqrt{x^2 + x} - x\right) = \dfrac{1}{2}$",
    fontsize=14,
)
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$y$", fontsize=12)
plt.xlim(0, x[-1])
plt.ylim(0, 0.75)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.1f}"))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.25))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.2f}"))
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.9)
plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.8)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()
