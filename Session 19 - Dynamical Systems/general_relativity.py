#!/usr/bin/env -S uv run
"""general_relativity.py

Estimate Mercury's GR perihelion precession rate
by sweeping the modified gravity correction alpha
and extrapolating to the physical Schwarzschild value
"""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

# Physical constants (SI → AU/yr)
m_sun = 1.98847e30
G = 6.67430e-11
year_s = 3.15576e7
AU = 1.495978707e11
GM = G * m_sun * year_s**2 / AU**3  # AU^3 / yr^2


def accel(xv, yv, alpha):
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
    alpha : float
        GR force correction coefficient (AU²); scales the strength of the
        1/r^4 perturbation term

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


def estimate_precession_slope(alpha):
    """
    Simulate Mercury's orbit with a given general relativity correction alpha
    and return the slope of perihelion angle vs time (degrees per year).

    Uses the Yoshida 4th-order symplectic integrator for accurate energy
    conservation, which reduces bias in the perihelion angle slope estimate.

    Parameters
    ----------
    alpha : float
        Modified gravity force correction to induce precession (AU²)

    Returns
    -------
    float
        Slope of perihelion angle vs time (degrees per year)
    """
    tf = 2  # final time in Julian years
    n = ceil(tf * 365.25 * 24)  # sample every hour
    dt = tf / n  # time step in Julian years

    # Mercury semi-major axis (AU) — matched to alpha correction scaling
    a_merc = 0.47

    # Start at perihelion on +x axis
    r_p = a_merc
    v_p = 8.2  # AU/yr

    # Time, radius(distance), and velocity arrays
    t = np.arange(n) * dt
    r = np.empty(n)
    x = np.empty(n)
    y = np.empty(n)
    vx = np.empty(n)
    vy = np.empty(n)

    # Mercury's orbital initial conditions (perihelion on +x axis)
    x[0] = r_p  # AU
    y[0] = 0.0
    vx[0] = 0.0
    vy[0] = v_p  # AU/yr (exact vis-viva at perihelion)
    r[0] = r_p

    # Yoshida 4th-order coefficients
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
            ax, ay = accel(px, py, alpha)
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

    # Fit line to perihelion angle vs time to get slope
    m, _ = np.polyfit(t[peri_idx], peri_angles, 1)  # degree=1 → line
    # m = Slope of perihelion angle vs time (degrees per year)

    return m


def main():
    """Main function to run the simulation and plot results."""

    # Sweep over alpha values to determine slope of perihelion angle vs time
    alpha_span = np.linspace(0.0001, 0.001, 11)
    print(alpha_span)
    slopes = np.array([estimate_precession_slope(alpha) for alpha in tqdm(alpha_span)])

    # Fit line to slope vs alpha data
    m, b = np.polyfit(alpha_span, slopes, 1)  # degree=1 → line
    y_fit = m * alpha_span + b  # y = mx + b

    # Compute alpha_GR from first principles (Goldstein Ch.3 / Schwarzschild metric).
    # The GR correction adds an extra 1/r^4 force term equivalent to multiplying
    # the Newtonian force by (1 + alpha/r^2), where:
    #   alpha = 3 * GM * a * (1 - e^2) / c^2
    # We use the simulated orbit's actual elements (derived from a=0.47, vy=8.2)
    # since the slope m was measured from that orbit, not real Mercury's orbit.
    c_au_yr = 2.99792458e8 * year_s / AU  # speed of light (AU/yr)
    # Simulated orbit elements from vis-viva (r=0.47 AU, v=8.2 AU/yr tangential)
    r_sim, v_sim = 0.47, 8.2
    a_sim = 1.0 / (2.0 / r_sim - v_sim**2 / GM)
    h_sim = r_sim * v_sim  # specific angular momentum
    e_sim = np.sqrt(1.0 - h_sim**2 / (GM * a_sim))
    alpha_GR = 3 * GM * a_sim * (1 - e_sim**2) / c_au_yr**2  # ~1.1142e-8 AU^2
    precession = m * alpha_GR * 3_600 * 100  # arcseconds per century (m = slope)

    # Plot slope of perihelion angle vs time as a function of alpha
    fig, ax = plt.subplots(num=Path(__file__).name)
    ax.set_title(
        "Estimated Precession of Mercury = "
        rf"$\mathbf{{{precession:.4f}}}\ \mathrm{{arcsec/century}}$"
    )
    ax.scatter(alpha_span, slopes, marker="o", c="r", label="Simulation Data")
    ax.plot(alpha_span, y_fit, lw=2, label="Linear Fit")
    ax.set_xlabel(r"$\alpha$ (AU$^2$)")
    ax.set_ylabel(r"$\dfrac{d\theta}{dt}$ (degrees/yr)")
    ax.legend(loc="center right", framealpha=1.0, facecolor="white")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
