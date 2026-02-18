#!/usr/bin/env -S uv run
"""plot3d_monolith.py"""

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

length = 40  # X direction
width = 10  # Y direction
height = 90  # Z direction

vertices: list = [tuple] * 8
vertices[0] = (0, 0, 0)  # Front Left Bottom
vertices[1] = (length, 0, 0)  # Front Right Bottom
vertices[2] = (length, width, 0)  # Back Right Bottom
vertices[3] = (0, width, 0)  # Back Left Bottom

vertices[4] = (0, 0, height)  # Front Left Top
vertices[5] = (length, 0, height)  # Front Right Top
vertices[6] = (length, width, height)  # Back Right Top
vertices[7] = (0, width, height)  # Back Left Top

facets: list = [tuple] * 6
facets[0] = (vertices[0], vertices[1], vertices[2], vertices[3])  # Bottom
facets[1] = (vertices[4], vertices[5], vertices[6], vertices[7])  # Top
facets[2] = (vertices[0], vertices[4], vertices[7], vertices[3])  # Left
facets[3] = (vertices[1], vertices[2], vertices[6], vertices[5])  # Right
facets[4] = (vertices[0], vertices[1], vertices[5], vertices[4])  # Front
facets[5] = (vertices[2], vertices[3], vertices[7], vertices[6])  # Back

p = Poly3DCollection(facets, linewidth=3, edgecolors=["Blue"], facecolors=["None"])

plt.figure(Path(__file__).name)
ax = plt.axes(projection="3d")
ax.view_init(azim=-45)
ax.add_collection3d(p)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim3d(xmin=-100, xmax=100)
ax.set_ylim3d(ymin=-100, ymax=100)
ax.set_zlim3d(zmin=0, zmax=100)
plt.show()
