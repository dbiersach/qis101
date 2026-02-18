#!/usr/bin/env -S uv run
"""rc_circuit.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import (
    AutoMinorLocator,
    FormatStrFormatter,
    MaxNLocator,
    MultipleLocator,
)

# Read samples
file_name = "rc_samples.csv"
file_path = Path(__file__).parent / file_name
times, volts = np.genfromtxt(file_path, delimiter=",", unpack=True)

# Set times to be elapsed time & scale to seconds
times -= times[0]
times *= 1e-9

# Find the middle time value
mid_time = times[-1] / 2

# Scale volts to fall between 0 and 3.3V
volts /= 65535
volts *= 3.3

# Calculate theoretical performance curve
V_s = 3.3  # Volts
R = 10_121  # Ohms
C = 0.00001069  # Farads
tau = R * C
t = np.linspace(0, mid_time, 100)
v1_c = V_s * (1 - np.exp(-t / tau))  # Charge
v2_c = V_s * np.exp(-t / tau)  # Decay

# Create a plot window
plt.figure(Path(__file__).name)
plt.gca().set_facecolor("black")

# Plot theoretical voltage
plt.plot(t, v1_c, color="cyan", linewidth=2, label="Theory")
plt.plot(t + mid_time, v2_c, color="cyan", linewidth=2)

# Plot actual voltage
# plt.plot(times, volts, color="magenta", linewidth=2, label="Actual")

# Give the graph a title, axis labels, and display the legend
plt.title("Capacitor Voltage vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# Set tick marks
plt.gca().xaxis.set_major_locator(MaxNLocator(11))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
plt.grid(which="both", color="gray", linestyle="dotted", alpha=0.5)

# Create straight lines to depict charging and discharging periods
line_on = [(0, 0), (0, 3.3)]
line_charging = [(0, 3.3), (mid_time, 3.3)]
line_off = [(mid_time, 3.3), (mid_time, 0)]
line_discharging = [(mid_time, 0), (times[-1], 0)]
lc = LineCollection(
    [line_on, line_charging, line_off, line_discharging],
    color="yellow",
    linewidth=2,
    zorder=2.5,
)
plt.gca().add_collection(lc)

# Draw dashed lines starting at offset 0, with 5 units "on" and 6 units "off"
plt.axvline(0.1, color="yellow", linestyle=(0, (5, 6)), alpha=0.65)
plt.axvline(mid_time + 0.1, color="yellow", linestyle=(0, (5, 6)), alpha=0.65)

plt.show()
