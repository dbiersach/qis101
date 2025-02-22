#!/usr/bin/env python3
"""plot_liquid_range.py"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Create a Python dictionary from the JSON file
file_name = "periodic_table.json"
file_path = Path(__file__).parent / file_name
with open(file_path, "r", encoding="utf-8") as infile:
    periodic_table = json.load(infile)

# Create list of elements sorted by group, period, atomic number
elements: list = []
for group in range(1, 19):
    for period in range(1, 8):
        for k, v in enumerate(periodic_table["elements"]):
            if group == int(v["group"]) and period == int(v["period"]):
                elements.append(
                    (
                        f"{v['symbol']}-{v['number']}",
                        float(v["melt"] or -np.inf),
                        float(v["boil"] or np.inf),
                    )
                )

# Create numpy arrays from sorted elements list
data = np.array(elements)
melt = np.array(data[:, 1], dtype=float) - 273.15
boil = np.array(data[:, 2], dtype=float) - 273.15

# Find element with smallest liquid range
liquid_range = boil - melt
min_idx = np.argmin(liquid_range)
min_range = liquid_range[min_idx]
print(f"Smallest liquid range: {min_range:>8,.2f}°C is {data[min_idx, 0]}")

# Find element with largest liquid range
measured_liquid_range = np.array(sorted(liquid_range, reverse=True))
max_measured_idx = np.argwhere(measured_liquid_range < np.inf).min()
max_range = measured_liquid_range[max_measured_idx]
max_idx = np.argwhere(liquid_range == max_range)[0, 0]
print(f" Largest liquid range: {max_range:>8,.2f}°C is {data[max_idx, 0]}")

# Plot the melting and boiling point of every element
plt.figure(Path(__file__).name, constrained_layout=True)
plt.get_current_fig_manager().full_screen_toggle()
x = np.arange(len(elements))
plt.plot(x, melt, color="turquoise", marker=".", label="Melting Point")
plt.plot(x, boil, color="coral", marker=".", label="Boiling Point")
plt.title("Melting and Boiling Point")
plt.xlabel("Elements (By Group, Period, Atomic Number)")
plt.ylabel("Temperature (C)")
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(data[:, 0], fontsize=9, rotation=90)
ax.legend(loc="lower center")
ax.grid("On")
plt.show()
