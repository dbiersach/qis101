#!/usr/bin/env python3
"""welcome.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 500)
f_top = np.sqrt(1 - (np.abs(x) - 1) ** 2)
f_bottom = np.arccos(1 - np.abs(x)) - np.pi

plt.figure(Path(__file__).name)
plt.plot(x, f_top, color="red")
plt.plot(x, f_bottom, color="red")
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 1.5)
plt.title("Welcome to QIS101")
plt.show()
