#!/usr/bin/env -S uv run
"""sphere_sampling.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Create random samples within the sphere
n = 15000
u = np.random.rand(n) * np.pi  # poloidal angle
v = np.random.rand(n) * 2 * np.pi  # toroidal angle

# Spherical to Cartesian coordinate conversion
x = np.sin(u) * np.sin(v)
y = np.sin(u) * np.cos(v)
z = np.cos(u)

# add_subplot() is used instead of plt.axes() because it is the call that
# actually hands back a 3D axes object, with methods such as view_init()
fig = plt.figure(Path(__file__).name, figsize=(10, 8))
ax = fig.add_subplot(projection="3d")
ax.view_init(azim=-72, elev=48)
# ax.plot() draws the same dots as ax.scatter(), and takes the z values
# as an ordinary third argument
ax.plot(x, y, z, ls="", marker="o", ms=0.7)
ax.set_title("Uniform Sphere Surface Sampling")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
