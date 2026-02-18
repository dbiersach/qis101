#!/usr/bin/env -S uv run
"""plot3d_pyramid_instructor.py"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

golden_ratio = (1 + math.sqrt(5)) / 2

length = 150  # X direction
width = 150  # Y direction
height = length * golden_ratio  # Z direction

vertices: list = [tuple] * 5
vertices[0] = (0, 0, 0)  # Base Front Left
vertices[1] = (length, 0, 0)  # Base Front Right
vertices[2] = (length, width, 0)  # Base Back Right
vertices[3] = (0, width, 0)  # Base Back Left
vertices[4] = (length / 2, width / 2, height)  # Apex

facets: list = [tuple] * 5
facets[0] = (vertices[0], vertices[1], vertices[2], vertices[3])  # Base
facets[1] = (vertices[0], vertices[3], vertices[4])  # Left
facets[2] = (vertices[0], vertices[1], vertices[4])  # Front
facets[3] = (vertices[1], vertices[2], vertices[4])  # Right
facets[4] = (vertices[2], vertices[3], vertices[4])  # Back

p = Poly3DCollection(
    facets, linewidth=3, edgecolors=["darkgoldenrod"], facecolors=["gold"]
)

plt.figure(Path(__file__).name)
ax = plt.axes(projection="3d")
ax.view_init(azim=-79, elev=28)
ax.add_collection3d(p)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim3d(xmin=-25, xmax=225)
ax.set_ylim3d(ymin=-25, ymax=225)
ax.set_zlim3d(zmin=0, zmax=350)
plt.show()
