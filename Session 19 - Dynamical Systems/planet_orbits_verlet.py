#!/usr/bin/env -S uv run
"""planet_orbits_verlet.py

Simulate heliocentric 2D *two-body* planet orbits (each planet around the Sun)
under Newtonian inverse-square gravity.

- Uses **Velocity Verlet** (symplectic 2nd-order integrator).
- Estimates orbital period using **unwrapped true anomaly** (robust for near-circular orbits).
- Estimates eccentricity using the **eccentricity (Laplace-Runge-Lenz) vector**:
    e_vec = (v × h)/mu - r_hat
  which is far less sensitive than r_min/r_max when e is small (e.g., Venus).

Units:
- AU for distance
- Julian years (365.25 d) for time
"""

from math import ceil, tau
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------
tf = 400.0  # final time in Julian years (365.25 days)
steps_per_day = 4  # 4 => dt = 6 hours
n = ceil(tf * 365.25 * steps_per_day)
dt = tf / n  # time step (years)

# -----------------------------------------------------------------------------
# Physical constants (SI -> AU/yr)
# -----------------------------------------------------------------------------
m_Sun = 1.98847e30  # kg
G = 6.67430e-11  # m^3 kg^-1 s^-2
year_in_seconds = 3.15576e7  # s (Julian year)
AU = 1.495978707e11  # m
mu = G * m_Sun * (year_in_seconds**2) / (AU**3)  # AU^3 / yr^2  (GM_sun)

# -----------------------------------------------------------------------------
# Planet data: (Name, Color, a [AU], e [-])
# e values are standard reference values (J2000-era; NASA Planet Compare table).
# -----------------------------------------------------------------------------
planets = [
    ("Venus", "peru", 0.723332, 0.00677672),
    ("Earth", "dodgerblue", 1.000000, 0.01671123),
    ("Mars", "red", 1.523679, 0.09339410),
    ("Jupiter", "chocolate", 5.2044, 0.04838624),
    ("Saturn", "gold", 9.5826, 0.05386179),
    ("Uranus", "mediumturquoise", 19.2184, 0.04725744),
    ("Neptune", "royalblue", 30.1104, 0.00859048),
]
planet_count = len(planets)

a_au = np.array([p[2] for p in planets], dtype=float)
e_ref = np.array([p[3] for p in planets], dtype=float)

# -----------------------------------------------------------------------------
# State arrays
# -----------------------------------------------------------------------------
t = np.arange(n, dtype=float) * dt  # years
x = np.zeros((n, planet_count), dtype=float)  # AU
y = np.zeros((n, planet_count), dtype=float)  # AU
vx = np.zeros((n, planet_count), dtype=float)  # AU/yr
vy = np.zeros((n, planet_count), dtype=float)  # AU/yr

# -----------------------------------------------------------------------------
# Initial conditions: start at perihelion on +x axis.
#   r_p = a (1 - e)
#   v_p = sqrt( mu * (1 + e) / (a (1 - e)) )  (vis-viva at perihelion)
# -----------------------------------------------------------------------------
r_p = a_au * (1.0 - e_ref)
v_p = np.sqrt(mu * (1.0 + e_ref) / (a_au * (1.0 - e_ref)))

x[0, :] = r_p
y[0, :] = 0.0
vx[0, :] = 0.0
vy[0, :] = v_p


def accel(xv: np.ndarray, yv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return acceleration components for arrays of x,y (AU) under -mu r/r^3."""
    r = np.sqrt(xv * xv + yv * yv)
    inv_r3 = 1.0 / (r * r * r)
    return -mu * xv * inv_r3, -mu * yv * inv_r3


# -----------------------------------------------------------------------------
# Integrate using Velocity Verlet (drift-kick-drift)
# -----------------------------------------------------------------------------
ax0, ay0 = accel(x[0, :], y[0, :])

for i in tqdm(range(1, n)):
    # Drift: update positions using current velocity and acceleration
    x[i, :] = x[i - 1, :] + vx[i - 1, :] * dt + 0.5 * ax0 * dt * dt
    y[i, :] = y[i - 1, :] + vy[i - 1, :] * dt + 0.5 * ay0 * dt * dt

    # Compute new acceleration at updated position
    ax1, ay1 = accel(x[i, :], y[i, :])

    # Kick: update velocities using average acceleration
    vx[i, :] = vx[i - 1, :] + 0.5 * (ax0 + ax1) * dt
    vy[i, :] = vy[i - 1, :] + 0.5 * (ay0 + ay1) * dt

    ax0, ay0 = ax1, ay1


# -----------------------------------------------------------------------------
# Helpers: period via unwrapped anomaly; eccentricity via LRL e-vector
# -----------------------------------------------------------------------------
def e_vector(
    xv: np.ndarray, yv: np.ndarray, vxv: np.ndarray, vyv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eccentricity vector magnitude for 2D motion around central mu."""
    r = np.sqrt(xv * xv + yv * yv)
    h = xv * vyv - yv * vxv  # specific angular momentum (z-component)
    ex = (vyv * h) / mu - (xv / r)  # (v × h) in-plane: vy*h
    ey = (-vxv * h) / mu - (yv / r)  # (v × h) in-plane: -vx*h
    return ex, ey, np.sqrt(ex * ex + ey * ey)


print("\nPlanet Orbits — Kepler's Third Law - Verlet 2nd-Order Symplectic Integrator:")
for j in range(planet_count):
    name, _, a, e0 = planets[j]

    th = np.unwrap(np.arctan2(y[:, j], x[:, j]))

    k = np.searchsorted(th, th[0] + tau)
    if k <= 1 or k >= len(th):
        print(f"{name:>9}: insufficient span to estimate period/eccentricity")
        continue

    T = t[k] - t[0]
    ratio = T**2 / a**3

    _, _, e_series = e_vector(
        x[: k + 1, j], y[: k + 1, j], vx[: k + 1, j], vy[: k + 1, j]
    )
    ehat = float(np.mean(e_series))
    de = ehat - e0
    pct = (de / e0 * 100.0) if e0 != 0 else np.nan

    r = np.sqrt(x[:, j] ** 2 + y[:, j] ** 2)
    v2 = vx[:, j] ** 2 + vy[:, j] ** 2
    eps = 0.5 * v2 - mu / r
    eps_drift = np.max(np.abs((eps - eps[0]) / eps[0]))

    pct = 0.0 if pct == 0.0 else pct
    print(
        f"{name:>9}: T= {T:>8.4f} yr, a= {a:>7.4f} AU, T^2/a^3= {ratio:.6f} yr^2/AU^3"
        f"  e= {ehat:>9.7f} ({pct:+6.4f}%)"
        f"  max drift: |ΔE/E|={eps_drift:.2e}"
    )
print()

# -----------------------------------------------------------------------------
# Plot: planet orbits
# -----------------------------------------------------------------------------
plt.figure(Path(__file__).name, figsize=(10, 10))
plt.gca().set_facecolor("black")
for j in range(planet_count):
    name, color, _, _ = planets[j]
    plt.plot(x[:, j], y[:, j], c=color, label=name)
plt.scatter([0.0], [0.0], s=20, marker="o", c="yellow", label="Sun")
plt.title(f"Planet Orbits — Velocity Verlet ({tf:g} Julian Earth Years)")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend(loc="upper right", framealpha=1.0)
plt.tight_layout()
plt.show()
