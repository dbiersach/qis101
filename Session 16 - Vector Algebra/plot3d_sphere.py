#!/usr/bin/env -S uv run
"""plot3d_sphere.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

u = np.linspace(0, np.pi, 30)  # poloidal angle
v = np.linspace(0, 2 * np.pi, 30)  # toroidal angle

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

plt.figure(Path(__file__).name)
ax = plt.axes(projection="3d")

ax.scatter(x, y, z)
# ax.plot_wireframe(x, y, z)
# ax.plot_surface(x, y, z)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.show()
