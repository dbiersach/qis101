#!/usr/bin/env -S uv run
"""limit_d_exp.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Left plot: limit of the difference quotient
x1 = np.linspace(-3, 3, 1000)
y1 = (np.exp(x1) - 1) / x1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), num=Path(__file__).name)
ax1.plot(x1, y1, "b-", linewidth=1.5, label=r"$f(x) = \dfrac{e^x - 1}{x}$")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.axvline(0, color="black", linewidth=0.5)
ax1.plot(0, 1, "ro", markersize=8, label=r"$\lim_{x \to 0} \dfrac{e^x - 1}{x} = 1$")
ax1.axhline(1, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax1.set_title(
    r"$\lim_{x \to 0} \dfrac{e^x - 1}{x} = 1$",
    fontsize=13,
)
ax1.set_xlabel(r"$x$", fontsize=12)
ax1.set_ylabel(r"$y$", fontsize=12)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-2, 4)
ax1.set_aspect("equal")
ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.2f}"))
ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.2f}"))
ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
ax1.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.9)
ax1.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.8)
ax1.legend(loc="upper left", fontsize=10)

# Right plot: e^x with tangent line at x=0
x2 = np.linspace(-3, 3, 1000)
y2 = np.exp(x2)
tangent = x2 + 1  # slope=1 → 45° when axes are equal

ax2.plot(x2, y2, "b-", linewidth=1.5, label=r"$f(x) = e^x$")
ax2.plot(x2, tangent, "r-", lw=1.5, label="tangent line")
ax2.plot(0, 1, "ro", markersize=8)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.axvline(0, color="black", linewidth=0.5)

ax2.set_title(r"$e^x$ and its Tangent Line at $x = 0$", fontsize=13)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel(r"$y$", fontsize=12)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-2, 4)
ax2.set_aspect("equal")  # equal axes → tangent visually appears at 45°
ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.2f}"))
ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.2f}"))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
ax2.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.9)
ax2.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.8)
ax2.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.show()
