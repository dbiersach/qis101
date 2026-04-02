#!/usr/bin/env -S uv run
"""planet_orbits_cromer.py"""

# Simulate heliocentric 2D orbits (Newtonian inverse-square gravity)
# for Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Simulation settings
tf = 400  # final time in Julian years (365.25 days)
n = ceil(tf * 365.25 * 2)  # number of time steps (2 per day)
dt = tf / n  # time step (12 hours)

# Physical constants (SI -> AU/yr)
m_Sun = 1.98847e30  # (kg)
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
year_in_seconds = 3.15576e7  # (s)
AU = 1.495978707e11  # Mean Earth-Sun distance (m)
GM = G * m_Sun * (year_in_seconds**2) / (AU**3)  # (AU^3 / yr^2)

# Planet data: Name, Color, Orbital ellipse semi-major axis "a" (AU)
planets = [
    ("Venus", "peru", 0.723332),
    ("Earth", "dodgerblue", 1.000000),
    ("Mars", "red", 1.523679),
    ("Jupiter", "chocolate", 5.2044),
    ("Saturn", "gold", 9.5826),
    ("Uranus", "mediumturquoise", 19.2184),
    ("Neptune", "royalblue", 30.1104),
]
planet_count = len(planets)

# Create numpy array to hold planet's orbital semi-major "a" radius (AU)
a_au = np.array([p[2] for p in planets], dtype=float)

# Time, distance, and velocity arrays
t = np.arange(n, dtype=float) * dt  # (Julian years)
x = np.zeros((n, planet_count), dtype=float)  # (AU)
y = np.zeros((n, planet_count), dtype=float)  # (AU)
vx = np.zeros((n, planet_count), dtype=float)  # (AU/yr)
vy = np.zeros((n, planet_count), dtype=float)  # (AU/yr)

# Initial conditions
x[0, :] = a_au  # Major axis circular orbit position
y[0, :] = 0.0  # Minor axis circular orbit position
vx[0, :] = 0.0  # Major axis circular orbit velocity
vy[0, :] = np.sqrt(GM / a_au)  # Minor axis circular orbit velocity

# Numerically estimate linked first-order differential equations
for i in tqdm(range(1, n)):
    r = np.sqrt(x[i - 1, :] ** 2 + y[i - 1, :] ** 2)
    ax = -GM * x[i - 1, :] / r**3
    ay = -GM * y[i - 1, :] / r**3
    vx[i, :] = vx[i - 1, :] + ax * dt
    vy[i, :] = vy[i - 1, :] + ay * dt
    x[i, :] = x[i - 1, :] + vx[i, :] * dt  # Euler-Cromer update
    y[i, :] = y[i - 1, :] + vy[i, :] * dt  # Euler-Cromer update

# Demonstrate Kepler's Third Law: T^2 proportional to a^3
print("\nPlanet Orbits — Kepler's Third Law - Euler-Cromer Integrator:")
for j in range(planet_count):
    name, _, a = planets[j]
    # Detect successive outward crossings of r ≈ a (numerical radius oscillations)
    # to estimate one full orbital period in a coordinate-independent way
    r = np.sqrt(x[:, j] ** 2 + y[:, j] ** 2)
    crossing_indices = np.where((r[:-1] < a) & (r[1:] >= a))[0]
    # Calculate the Orbital period (years) between two consecutive crossings
    T = t[crossing_indices[1]] - t[crossing_indices[0]]
    # Kepler's ratio T^2/a^3, should be constant for all planets
    # In AU units, this ratio is approximately 1 (yr^2/AU^3)
    ratio = T**2 / a**3
    print(
        f"{name:>9}: T= {T:>8.4f} yr, a= {a:>7.4f} AU, T^2/a^3= {ratio:.6f} yr^2/AU^3"
    )
print()

# Plot planet orbits
plt.figure(Path(__file__).name, figsize=(10, 10))
plt.gca().set_facecolor("black")
for j in range(planet_count):
    name, color, _ = planets[j]
    plt.plot(x[:, j], y[:, j], c=color, label=name)
plt.scatter([0.0], [0.0], s=20, marker="o", c="yellow", label="Sun")
plt.title(f"Planet Orbits — Euler-Cromer ({tf} Julian Earth Years)")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend(framealpha=1.0)
plt.tight_layout()
plt.show()
