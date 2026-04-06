#!/usr/bin/env -S uv run
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
elements: list = sorted(
    periodic_table["elements"],
    key=lambda v: (int(v["group"]), int(v["period"]), int(v["number"])),
)
elements = [
    (
        f"{v['symbol']}-{v['number']}",
        float(v["melt"]) if v["melt"] is not None else -np.inf,
        float(v["boil"]) if v["boil"] is not None else np.inf,
    )
    for v in elements
]

# Create numpy arrays from sorted elements list
data = np.array(elements)
melt = np.array(data[:, 1], dtype=float) - 273.15
boil = np.array(data[:, 2], dtype=float) - 273.15

# Find element with smallest liquid range
liquid_range = boil - melt
min_idx = np.argmin(liquid_range)
min_range = liquid_range[min_idx]
print(f"Smallest liquid range: {min_range:>8,.2f}°C is {data[min_idx, 0]}")

# Find element with largest liquid range (excluding elements with unknown boil point)
finite_mask = np.isfinite(liquid_range)
max_idx = np.argmax(np.where(finite_mask, liquid_range, -np.inf))
max_range = liquid_range[max_idx]
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
ax.grid(True)
plt.show()
