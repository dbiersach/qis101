#!/usr/bin/env -S uv run
"""plot3d_helix.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 20 * np.pi, 2000)  # poloidal angle
x = theta * np.cos(theta)
y = theta * np.sin(theta)
z = theta

# add_subplot() is used instead of plt.axes() because it is the call that
# actually hands back a 3D axes object, with methods such as set_zlabel()
fig = plt.figure(Path(__file__).name)
ax = fig.add_subplot(projection="3d")
ax.view_init(azim=-45)
ax.plot(x, y, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
