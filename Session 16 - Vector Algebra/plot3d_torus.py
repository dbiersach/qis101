#!/usr/bin/env -S uv run
"""plot3d_torus.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

radius_poloidal = 5
radius_toroidal = 25

u = np.linspace(0, 2 * np.pi, 60)  # Poloidal rotation
v = np.linspace(0, 2 * np.pi, 60)  # Toroidal rotation

x = np.outer(radius_toroidal + radius_poloidal * np.sin(u), np.cos(v))
y = np.outer(radius_toroidal + radius_poloidal * np.sin(u), np.sin(v))
z = np.outer(radius_poloidal * np.cos(u), np.ones_like(v))

# add_subplot() is used instead of plt.axes() because it is the call that
# actually hands back a 3D axes object, with methods such as set_zlabel()
fig = plt.figure(Path(__file__).name)
ax = fig.add_subplot(projection="3d")
ax.view_init(azim=132, elev=-144)

# ax.plot() draws the same dots as ax.scatter(), and takes the z values
# as an ordinary third argument. ravel() flattens the grids into point lists
ax.plot(x.ravel(), y.ravel(), z.ravel(), ls="", marker="o", ms=4.5, color="gold")
# ax.plot_surface(x, y, z, rcount=60, ccount=60, color="gold")

ax.set_xlim(-radius_toroidal, radius_toroidal)
ax.set_ylim(-radius_toroidal, radius_toroidal)
ax.set_zlim(-radius_toroidal, radius_toroidal)

ax.set_aspect("equal")
plt.show()
