#!/usr/bin/env -S uv run
"""plot_field_strength.py"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from numpy.polynomial import Polynomial

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
c2 = Polynomial.fit(dist[1:], field_strength[1:], 2).convert()
c3 = Polynomial.fit(dist[1:], field_strength[1:], 3).convert()

# Create smooth arrays to store estimated values
est_x = np.linspace(2, 22, 500)
est_y2 = c2(est_x)
est_y3 = c3(est_x)

dist_0, act_0 = dist[0], field_strength[0]
c2_0, c3_0 = c2(dist_0), c3(dist_0)
err2 = abs((act_0 - c2_0) / act_0)
err3 = abs((act_0 - c3_0) / act_0)

print(f"Measured field strength at {dist_0} cm   : {act_0:,}")
print(f"Est. (quadratic) strength at {dist_0} cm : {c2_0:,.2f}")
print(f"Est. (cubic) strength at {dist_0} cm     : {c3_0:,.2f}")
print(f"Absolute percent error (quadratic)  : {err2:.2%}")
print(f"Absolute percent error (cubic)      : {err3:.2%}")

# Plot the sample data and the two polynomials
plt.figure(Path(__file__).name)
plt.scatter(dist, field_strength, color="black", label="Measured")
plt.plot(est_x, est_y2, color="red", label="Quadratic")
plt.plot(est_x, est_y3, color="blue", label="Cubic")
plt.legend()
plt.title("Magnetic Dipole Field Strength vs. Distance")
plt.xlabel("Distance (cm)")
plt.ylabel("Field Strength (uT)")
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.show()
