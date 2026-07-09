#!/usr/bin/env -S uv run
"""half_step_diagram.py

Generates half_step.png, the figure embedded in mc_circle_grid.ipynb.
Explains why the grid linspace runs from -1 + half_step to 1 - half_step.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

n = 8  # stands in for side_dots (320), kept coarse so the cells are legible
h = 1 / n  # half_step
edges = np.linspace(-1, 1, n + 1)  # cell boundaries
centers = np.linspace(-1 + h, 1 - h, n)  # midpoint rule (what the notebook uses)
naive = np.linspace(-1, 1, n)  # spacing 2/(n-1), endpoints on the boundary

fig, (ax_bad, ax_good) = plt.subplots(2, 1, figsize=(8.5, 4.4), num=Path(__file__).name)

for ax in (ax_bad, ax_good):
    for i in range(0, n, 2):  # shade alternate cells
        ax.axvspan(edges[i], edges[i + 1], color="0.93", zorder=0)
    for e in edges:
        ax.axvline(e, color="0.65", lw=0.8, zorder=1)
    ax.axhline(0, color="black", lw=1.2, zorder=2)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.75, 0.85)
    ax.set_yticks([])
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["$-1$", "$0$", "$1$"])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)

# Top panel: the naive linspace, whose points drift off the cell centers
ax_bad.set_title(
    r"$\mathtt{np.linspace(-1,\ 1,\ side\_dots)}$"
    "  —  spacing is $2/(n{-}1)$, so points drift off their cell centers",
    fontsize=10,
    pad=8,
)
ax_bad.scatter(naive, np.zeros(n), s=55, color="tab:blue", zorder=3)
ax_bad.scatter(
    [-1, 1], [0, 0], s=150, facecolors="none", edgecolors="tab:red", lw=1.4, zorder=4
)
for point, center in zip(naive, centers, strict=True):
    if abs(point - center) > 1e-9:  # red segment showing the drift
        ax_bad.annotate(
            "",
            xy=(point, 0),
            xytext=(center, 0),
            arrowprops={"arrowstyle": "-", "color": "tab:red", "lw": 1.0, "alpha": 0.7},
        )
    ax_bad.plot([center], [0], marker="|", ms=9, color="0.45", zorder=2)
ax_bad.annotate(
    "endpoints sit on the domain boundary",
    xy=(-1, 0),
    xytext=(-0.97, 0.55),
    fontsize=8.5,
    color="tab:red",
    arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.0},
)
ax_bad.annotate(
    "",
    xy=(1, 0),
    xytext=(0.60, 0.50),
    arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.0},
)
ax_bad.text(
    0.0,
    -0.52,
    "grey ticks = true cell centers;  red segments = the drift",
    ha="center",
    fontsize=8.5,
    color="0.35",
)

# Bottom panel: the half_step inset, one point per cell center
ax_good.set_title(
    r"$\mathtt{np.linspace(-1 + half\_step,\ 1 - half\_step,\ side\_dots)}$"
    "  —  exactly one point per cell center",
    fontsize=10,
    pad=8,
)
ax_good.scatter(centers, np.zeros(n), s=55, color="tab:green", zorder=3)

y_cell = 0.30  # cell-width bracket over the 4th cell
ax_good.annotate(
    "",
    xy=(edges[4], y_cell),
    xytext=(edges[3], y_cell),
    arrowprops={"arrowstyle": "<->", "color": "tab:orange", "lw": 1.4},
)
ax_good.text(
    (edges[3] + edges[4]) / 2,
    y_cell + 0.07,
    "cell width = 2/side_dots\n= 2 × half_step",
    ha="center",
    va="bottom",
    fontsize=8.5,
    color="tab:orange",
    linespacing=1.3,
)

y_arrow = -0.28  # half_step brackets at both ends
for x0, x1 in ((-1, -1 + h), (1 - h, 1)):
    ax_good.annotate(
        "",
        xy=(x1, y_arrow),
        xytext=(x0, y_arrow),
        arrowprops={"arrowstyle": "<->", "color": "tab:purple", "lw": 1.5},
    )
ax_good.text(
    -1 + h,
    y_arrow - 0.12,
    "half_step = 1/side_dots",
    ha="left",
    va="top",
    fontsize=8.5,
    color="tab:purple",
)
ax_good.text(
    1 - h,
    y_arrow - 0.12,
    "half_step",
    ha="right",
    va="top",
    fontsize=8.5,
    color="tab:purple",
)

fig.suptitle(
    f"Why the grid is inset by half_step   (drawn with side_dots = {n}; "
    "the notebook uses 320)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(Path(__file__).with_name("half_step.png"), dpi=150)
plt.show()
