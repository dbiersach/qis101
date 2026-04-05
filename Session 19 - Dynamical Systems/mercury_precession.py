#!/usr/bin/env -S uv run
"""mercury_precession.py

Simulate Mercury's heliocentric 2D orbit
with a modified inverse-square gravity term
to visualize perihelion precession
"""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Simulation settings
# alpha = Modified gravity force exponent to induce precession (AU²)
# tf = final time in Julian years
alpha, tf = 0.01, 2

# Physical constants (SI → AU/yr)
m_sun = 1.98847e30
G = 6.67430e-11
year_s = 3.15576e7
AU = 1.495978707e11
GM = G * m_sun * year_s**2 / AU**3  # AU^3 / yr^2


def accel(xv, yv):
    """
    Compute the GR-corrected gravitational acceleration on Mercury.

    Applies a modified inverse-square force law that multiplies the Newtonian
    acceleration by (1 + alpha/r^2), adding a 1/r^4 perturbation term that
    mimics the leading-order general relativity correction to Mercury's orbit.

    Parameters
    ----------
    xv : float
        x-coordinate of Mercury's position (AU)
    yv : float
        y-coordinate of Mercury's position (AU)

    Returns
    -------
    float
        x-component of acceleration (AU/yr²)
    float
        y-component of acceleration (AU/yr²)
    """
    rv = np.hypot(xv, yv)
    corr = 1.0 + alpha / rv**2
    ax = -GM * xv / rv**3 * corr  # x-component of acceleration (AU/yr²)
    ay = -GM * yv / rv**3 * corr  # y-component of acceleration (AU/yr²)
    return ax, ay


def main():
    a = 0.47  # Mercury semi-major axis (AU)

    n = ceil(tf * 365.25 * 24)  # Sample every hour
    dt = tf / n  # time step in Julian years

    # Time, radius(distance), and velocity arrays
    t = np.arange(n) * dt
    r = np.empty(n)
    x = np.empty(n)
    y = np.empty(n)
    vx = np.empty(n)
    vy = np.empty(n)

    # Mercury's orbital initial conditions
    x[0] = a  # AU
    y[0] = 0.0
    vx[0] = 0.0
    vy[0] = 8.2  # AU/yr
    r[0] = np.hypot(x[0], y[0])

    # Yoshida 4th-order symplectic integrator coefficients
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)
    c = np.array([w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0])
    d = np.array([w1, w0, w1])

    # Time integration (Yoshida 4th-order symplectic)
    for i in range(1, n):
        px, py = x[i - 1], y[i - 1]
        pvx, pvy = vx[i - 1], vy[i - 1]
        for j in range(3):
            px += c[j] * pvx * dt
            py += c[j] * pvy * dt
            ax, ay = accel(px, py)
            pvx += d[j] * ax * dt
            pvy += d[j] * ay * dt
        px += c[3] * pvx * dt
        py += c[3] * pvy * dt
        x[i], y[i] = px, py
        vx[i], vy[i] = pvx, pvy
        r[i] = np.hypot(x[i], y[i])

    # Calculate the perihelion points and angles
    peri_idx, _ = find_peaks(-r)
    peri_angles = np.degrees(np.unwrap((np.arctan2(y[peri_idx], x[peri_idx]))))

    # Plot orbit and perihelion precession
    plt.figure(Path(__file__).name, figsize=(8, 8))
    plt.gca().set_facecolor("black")
    # Draw the Sun at the origin
    plt.scatter([0.0], [0.0], c="yellow", s=40, label="Sun", zorder=3)
    # Draw Mercury's orbit
    plt.plot(x, y, c="silver", label="Mercury")
    # Draw perihelion markers, radial lines, and orbit numbers
    for orbit_num, k in enumerate(peri_idx, start=1):
        xk, yk = x[k], y[k]
        # Radial line
        plt.plot([0.0, xk], [0.0, yk], c="silver", lw=1.5, ls="--", alpha=0.5)
        # Plot and label Perihelion markers with orbit number
        plt.scatter([xk], [yk], c="silver", s=20, zorder=3)
        # fmt: off
        plt.text(
            xk * 1.01, yk * 1.01, f"{orbit_num}",
            color="red", fontsize=12, fontweight="bold",
            ha="left", va="bottom", zorder=4,
        )
        # fmt: on
    plt.title(rf"Mercury Perihelion Precession ({tf} Julian Years) $\alpha={alpha}$")
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(framealpha=1.0)
    plt.tight_layout()

    # Plot orbital distance and perihelion angle vs. time
    plt.figure("Orbital Distance vs. Time", figsize=(10, 4))
    plt.plot(t, r)
    plt.scatter(t[peri_idx], r[peri_idx], c="red", s=20, label="Perihelion")
    plt.title(rf"Orbital Distance vs. Time ($\alpha={alpha}$)")
    plt.xlabel("Time (Julian years)")
    plt.ylabel(r"Orbital Distance $r$ (AU)")
    plt.grid(True)
    plt.tight_layout()

    # Plot perihelion angle vs. time and linear fit
    plt.figure("Orbit Orientation vs. Time")
    plt.scatter(t[peri_idx], peri_angles, c="red")
    # Calculate and plot linear fit to perihelion angle vs time
    m, b = np.polyfit(t[peri_idx], peri_angles, 1)  # degree=1 → line
    y_fit = m * t[peri_idx] + b
    slope_label = (
        r"$\frac{d\Theta}{dt} = "
        rf"{m:.3f}\ \mathrm{{deg/yr}}$"
    )
    plt.plot(t[peri_idx], y_fit, lw=2, label="Linear Fit")
    # fmt: off
    plt.text(
        0.5, 0.5, slope_label,
        transform=plt.gca().transAxes,
        fontsize=14, ha="center", va="center",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    # fmt: on
    plt.title(rf"Orbital Orientation vs. Time ($\alpha={alpha}$)")
    plt.xlabel("Time (Julian years)")
    plt.ylabel(r"$\theta$ (degrees)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
