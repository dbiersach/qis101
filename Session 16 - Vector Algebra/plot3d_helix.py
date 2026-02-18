#!/usr/bin/env -S uv run
"""plot3d_helix.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 20 * np.pi, 2000)  # poloidal angle
x = theta * np.cos(theta)
y = theta * np.sin(theta)
z = theta

plt.figure(Path(__file__).name)
ax = plt.axes(projection="3d")
ax.view_init(azim=-45)
ax.plot(x, y, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
