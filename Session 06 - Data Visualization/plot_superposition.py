#!/usr/bin/env -S uv run
"""plot_superposition.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 4 * np.pi, 1000)
radius = 7 + 7 * np.sin(11 * theta) * np.cos(5 * theta)

plt.figure(Path(__file__).name)
plt.subplot(projection="polar")
# fmt:off
plt.plot(theta, radius, color="black",
    label=r"$7+7\,\sin{(11\,\theta)}\,\cos{(5\,\theta)}$")
# fmt:on
plt.legend(loc="upper right")
plt.axis(True)
plt.show()
