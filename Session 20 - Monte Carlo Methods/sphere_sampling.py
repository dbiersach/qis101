#!/usr/bin/env python3
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

plt.figure(Path(__file__).name, figsize=(10, 8))
ax = plt.axes(projection="3d")
ax.view_init(azim=-72, elev=48)
ax.scatter(x, y, z, s=0.5)
ax.set_title("Uniform Sphere Surface Sampling")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
