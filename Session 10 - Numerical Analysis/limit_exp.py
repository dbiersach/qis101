#!/usr/bin/env -S uv run
"""limit_exp.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 5, 1000)
# Create a Boolean mask to exclude the points at
# x=0 (singularity) and x=-1 (domain restriction)
x = x[(x != -1) & (x != 0)]
# Calculate (1+x)^(1/x)
y = (1 + x) ** (1 / x)

# Plot the function
plt.figure(Path(__file__).name, figsize=(10, 6))
plt.plot(x, y, "b-", linewidth=1.5, label=r"$f(x) = (1+x)^{1/x}$")

# Mark the limit point at x=0, y=e
plt.plot(
    0,
    np.e,
    "ro",
    markersize=8,
    label=rf"$\lim_{{x \to 0}} (1+x)^{{1/x}} = e \approx {np.e:.4f}$",
)
plt.axhline(np.e, color="red", linestyle="--", linewidth=1, alpha=0.5)

# Set title, labels, and limits
plt.title(
    r"The Limit Definition of $e$: $\lim_{x \to 0} (1+x)^{1/x} = e$",
    fontsize=14,
)
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$y$", fontsize=12)
plt.xlim(-1, 5)
plt.ylim(0, 4)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.25))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
plt.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.9)
plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.8)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()
