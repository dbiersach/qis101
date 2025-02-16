#!/usr/bin/env python3
"""plot_quintic.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 12)
y = (x - 11) * (x - 5) * (x + 1) * (x + 4) * (x + 9)

plt.figure(Path(__file__).name)
plt.plot(x, y)
plt.grid("on")
plt.show()
