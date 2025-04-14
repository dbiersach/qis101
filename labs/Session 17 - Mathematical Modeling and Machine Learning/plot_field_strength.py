#!/usr/bin/env python3
"""plot_field_strength.py"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Create a dictionary of existing sample readings
file_path = Path(__file__).parent / "field_strength.json"
if not file_path.exists():
    print(f'Cannot open "{file_path}"')
    # Indicate an error occurred by returning a non-zero exit code
    # Retrieve via 'echo $?' (bash) or 'echo $LASTEXITCODE' (pwsh)
    sys.exit(1)
with file_path.open("r") as file:
    samples = json.load(file)

# Create NumPy arrays to hold independent and dependent sample values
dist = np.array([float(k) for k in samples])
field_strength = np.array([float(v) for v in samples.values()])

# Use linear regression (least squares) to fit two polynomials
# Exclude from model the first data point (distance = 2cm)
c2 = np.polyfit(dist[1:], field_strength[1:], 2)  # quadratic coefficients
c3 = np.polyfit(dist[1:], field_strength[1:], 3)  # cubic coefficients

# Create smooth arrays to store estimated values
est_x = np.linspace(2, 22, 500)
est_y2 = c2[0] * est_x**2 + c2[1] * est_x + c2[2]
est_y3 = c3[0] * est_x**3 + c3[1] * est_x**2 + c3[2] * est_x + c3[3]

# Plot the sample data and the two polynomials
plt.figure(Path(__file__).name)
plt.scatter(dist, field_strength, color="black")
plt.plot(est_x, est_y2, color="red", label="Quadratic")
plt.plot(est_x, est_y3, color="blue", label="Cubic")
plt.legend()
plt.title("Magnetic Dipole Field Strength vs. Distance")
plt.xlabel("Distance (cm)")
plt.ylabel("Field Strength (uT)")
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.show()
