# plot_unknown_wave.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

alpha = 0
beta = 0

t = np.linspace(0, 2 * np.pi, 1000)
y = np.zeros_like(t)

plt.figure(Path(__file__).name)
plt.title("QIS102 Task 13-03: Unknown Wave")
plt.plot(t, y)
plt.grid("on")
plt.show()
