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

# add_subplot() is used instead of plt.axes() because it is the call that
# actually hands back a 3D axes object, with methods such as set_zlabel()
fig = plt.figure(Path(__file__).name)
ax = fig.add_subplot(projection="3d")

# ax.plot() draws the same dots as ax.scatter(), and takes the z values
# as an ordinary third argument. ravel() flattens the grids into point lists
ax.plot(x.ravel(), y.ravel(), z.ravel(), ls="", marker="o", ms=4.5)
# ax.plot_wireframe(x, y, z)
# ax.plot_surface(x, y, z)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.show()
