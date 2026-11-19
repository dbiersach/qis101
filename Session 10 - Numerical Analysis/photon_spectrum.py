#!/usr/bin/env -S uv run
"""photon_spectrum.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

file_name = "photon_spectrum.txt"
file_path = Path(__file__).parent / file_name

# For this exercise, treat column 2 as relative photon counts per MeV.
energy, density = np.genfromtxt(file_path, delimiter=" ", unpack=True)

min_energy, max_energy = np.min(energy), np.max(energy)
min_window, max_window = 2.12, 3.45  # MeV
density_f = CubicSpline(energy, density, extrapolate=False)

# Integrate the piecewise cubics over the window and the measured range.
window_count = density_f.integrate(min_window, max_window)
total_count = density_f.integrate(min_energy, max_energy)
window_fraction = window_count / total_count
print(f"{window_fraction:.4%} of photons in the sampled range are in the window.")

energy_est = np.linspace(min_energy, max_energy, 1000)
window_est = np.linspace(min_window, max_window, 200)
plt.figure(Path(__file__).name)
plt.scatter(energy, density, zorder=3)
plt.plot(energy_est, density_f(energy_est))
plt.fill_between(window_est, density_f(window_est), color="orange")
plt.xlabel("Photon Energy [MeV]")
plt.ylabel("Relative photon counts per MeV")
plt.title("Photon Spectrum")
plt.axhline(0, c="k")
plt.show()
