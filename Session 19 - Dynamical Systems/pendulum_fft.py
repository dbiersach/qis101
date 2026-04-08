#!/usr/bin/env -S uv run
"""pendulum_fft.py

Runs four integrators on an ideal pendulum for a long duration,
then compares their energy error and FFT power spectra to show how
spectral purity improves as the numerical method improves.

    1. RK45 (solve_ivp)  - not symplectic, high-order adaptive
    2. Euler-Cromer      - symplectic, 1st order
    3. Velocity Verlet   - symplectic, 2nd order
    4. Yoshida 4th-order - symplectic, 4th order

Each method gets its own figure window with two subplots:
    Left:  |Relative energy error| over time (log scale)
    Right: FFT power spectrum of angular displacement

Note: a sinusoid of amplitude A produces a spectral peak at exactly 20*log10(A) dB
For a 10° pendulum, the angular displacement oscillates between -10° and +10°
Therefore the amplitude is:
    A = 10° = 0.1745 rad
    20 * log10(0.1745) ≈ -15.2 dB (the peak in the power spectrum)

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from numpy.fft import fft
from scipy.integrate import solve_ivp

from qis101_utils import (
    PENDULUM_G,
    PENDULUM_LENGTH,
    pendulum_angular_acceleration,
    pendulum_euler_cromer,
    pendulum_total_energy,
    pendulum_velocity_verlet,
    pendulum_yoshida4,
)

# ========================================================================
# Integrators (all return t, theta, omega arrays)
# ========================================================================


def _ode_model(time, state_vector):
    """State-space form for solve_ivp."""
    omega, theta = state_vector
    return pendulum_angular_acceleration(theta), omega


def solve_rk45(theta0, omega0, t_final, dt):
    """
    Integrate the pendulum equations using SciPy's adaptive RK45 solver.

    Uses solve_ivp with a fixed maximum step size to ensure a uniform time
    grid comparable to the other integrators. RK45 is not symplectic, so
    energy is not conserved exactly over long integrations.

    Parameters
    ----------
    theta0 : float
        Initial angular displacement (radians)
    omega0 : float
        Initial angular velocity (rad/s)
    t_final : float
        Total integration time (s)
    dt : float
        Maximum step size and output sample interval (s)

    Returns
    -------
    t : ndarray
        Time array (s)
    theta : ndarray
        Angular displacement at each time step (radians)
    omega : ndarray
        Angular velocity at each time step (rad/s)
    """
    sol = solve_ivp(
        _ode_model,
        (0, t_final),
        [omega0, theta0],
        method="RK45",
        max_step=dt,
        dense_output=True,
    )
    t = np.arange(0, t_final, dt)
    y = sol.sol(t)
    return t, y[1], y[0]  # t, theta, omega


# ========================================================================
# Plotting helpers
# ========================================================================


def plot_energy_error(ax, t, abs_energy_error, color):
    """
    Plot the absolute relative energy error over time on a log scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the plot
    t : ndarray
        Time array (s)
    abs_energy_error : ndarray
        Absolute relative energy error |E − E₀| / |E₀| at each time step
    color : str
        Line color
    """
    ax.plot(t, abs_energy_error, color=color, linewidth=0.3)
    ax.set_yscale("log")
    ax.set_title("|Energy Error|")
    ax.set_xlabel("time (s)", loc="right")
    ax.set_ylabel("|E − E₀| / |E₀|")
    ax.grid(True, alpha=0.3, which="both")


def plot_power_spectrum(ax, freqs, power_db, color, f_max=3.0):
    """
    Plot the FFT power spectrum in dB up to a maximum frequency.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the plot
    freqs : ndarray
        Frequency axis (Hz)
    power_db : ndarray
        Power spectrum in decibels (dB)
    color : str
        Line color
    f_max : float, optional
        Upper frequency limit for the plot (Hz); default is 3.0
    """
    mask = (freqs > 0) & (freqs <= f_max)
    ax.plot(freqs[mask], power_db[mask], color=color, linewidth=0.8)

    ax.grid(which="major", axis="x", color="black", linewidth=1)
    ax.grid(which="minor", axis="x", color="lightgray", linewidth=1)
    ax.grid(which="major", axis="y", color="black", linewidth=1)
    ax.grid(which="minor", axis="y", color="lightgray", linewidth=1)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title("Power Spectrum")
    ax.set_xlabel("frequency (Hz)", loc="right")
    ax.set_ylabel("power (dB)")


def compute_spectrum(theta, dt):
    """
    Compute the normalized FFT power spectrum and frequency axis.

    Applies a two-sided-to-one-sided normalization (2/N scaling) so that
    spectral peak amplitudes match the time-domain signal amplitude. The DC
    component is not doubled. Power is returned both as linear amplitude
    and in decibels, with a noise floor clamp to avoid log(0).

    Parameters
    ----------
    theta : ndarray
        Angular displacement time series (radians)
    dt : float
        Sample interval (s)

    Returns
    -------
    freqs : ndarray
        Frequency axis (Hz)
    power : ndarray
        Linear amplitude spectrum (radians)
    power_db : ndarray
        Power spectrum in decibels (dB), clamped to a floor of 1e-15
    """
    n = len(theta)
    ct = 2 / n * fft(theta)
    ct[0] /= 2  # DC value should NOT be doubled
    freqs = np.arange(n) / (n * dt)  # frequency axis in Hz
    power = np.abs(ct)
    # Convert to dB (clamp floor to avoid log(0))
    power_db = 20 * np.log10(np.maximum(power, 1e-15))
    return freqs, power, power_db


def main():
    # --- Simulation parameters ---
    theta_initial = np.deg2rad(10)
    omega_initial = 0.0
    dt = 0.02  # 50 Hz sample rate
    t_final = 10000  # seconds - long run for spectral resolution

    E0 = pendulum_total_energy(theta_initial, omega_initial)

    # Expected fundamental (small-angle approximation for reference)
    f_natural = 1.0 / (2.0 * np.pi) * np.sqrt(PENDULUM_G / PENDULUM_LENGTH)

    methods = {
        "RK45 (solve_ivp)": (solve_rk45, "red"),
        "Euler-Cromer": (pendulum_euler_cromer, "orange"),
        "Velocity Verlet": (pendulum_velocity_verlet, "green"),
        "Yoshida 4th-Order": (pendulum_yoshida4, "blue"),
    }

    # Shared axis limits across all figures
    err_ylim = (1e-10, 1e0)
    db_min = -100
    db_max = 10

    results = {}
    for name, (solver, color) in methods.items():
        print(f"Running {name}...")
        t, theta, omega = solver(theta_initial, omega_initial, t_final, dt)
        E = pendulum_total_energy(theta, omega)
        abs_energy_error = np.maximum(np.abs((E - E0) / abs(E0)), 1e-16)
        freqs, power, power_db = compute_spectrum(theta, dt)
        results[name] = (t, abs_energy_error, freqs, power, power_db, color)

    # --- Create one figure per method ---
    for name in reversed(list(methods.keys())):
        t, abs_energy_error, freqs, power, power_db, color = results[name]

        fig, (ax_err, ax_fft) = plt.subplots(1, 2, figsize=(12, 5))
        fig_name = f"{Path(__file__).name} - {name}"
        fig.canvas.manager.set_window_title(fig_name)
        fig.suptitle(
            f"{name} [{t_final}s, dt={dt}s, "
            rf"$\theta_0$={np.rad2deg(theta_initial):.0f}°]"
            f"   (expected f ≈ {f_natural:.3f} Hz)",
            fontsize=13,
            fontweight="bold",
        )

        plot_energy_error(ax_err, t, abs_energy_error, color)
        ax_err.set_ylim(err_ylim)

        plot_power_spectrum(ax_fft, freqs, power_db, color)
        ax_fft.set_ylim(db_min, db_max)

        fig.tight_layout()

    # --- Print summary ---
    print(f"\nExpected fundamental ≈ {f_natural:.4f} Hz (small-angle approx)")
    print(
        f"{'Method':<22} {'Peak (Hz)':>10} {'Peak (dB)':>10}"
        f" {'Energy err std':>16} {'Energy err max':>16}"
    )
    print("-" * 78)
    for name in methods:
        t, abs_energy_error, freqs, power, power_db, color = results[name]
        mask = (freqs > 0.1) & (freqs < 2.0)
        idx_peak = np.argmax(power[mask])
        peak_f = freqs[mask][idx_peak]
        peak_db = power_db[mask][idx_peak]
        print(
            f"{name:<22} {peak_f:>10.4f} {peak_db:>10.1f}"
            f" {np.std(abs_energy_error):>16.2e}"
            f" {np.max(abs_energy_error):>16.2e}"
        )

    plt.show()


if __name__ == "__main__":
    main()
