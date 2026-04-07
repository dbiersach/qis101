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

# --- Physical constants ---
LENGTH = 1.0  # pendulum length (m)
G = 9.81  # gravity (m/s^2)


def angular_acceleration(theta):
    """
    Compute the angular acceleration of an ideal pendulum.

    Applies the exact (nonlinear) equation of motion: α = -(g/L) sin(θ),
    with no small-angle approximation.

    Parameters
    ----------
    theta : float or ndarray
        Angular displacement from vertical (radians)

    Returns
    -------
    float or ndarray
        Angular acceleration (rad/s²)
    """
    return -G / LENGTH * np.sin(theta)


def total_energy(theta, omega):
    """
    Compute the total mechanical energy of the pendulum per unit mass.

    Returns the sum of kinetic and potential energy in the form:
    E = (1/2)ω² - (g/L)cos(θ), where potential energy is referenced
    to the pivot point.

    Parameters
    ----------
    theta : float or ndarray
        Angular displacement from vertical (radians)
    omega : float or ndarray
        Angular velocity (rad/s)

    Returns
    -------
    float or ndarray
        Total mechanical energy per unit mass (J/kg)
    """
    return 0.5 * omega**2 - G / LENGTH * np.cos(theta)


# ========================================================================
# Integrators (all return t, theta, omega arrays)
# ========================================================================


def _ode_model(time, state_vector):
    """State-space form for solve_ivp."""
    omega, theta = state_vector
    return angular_acceleration(theta), omega


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


def solve_euler_cromer(theta0, omega0, t_final, dt):
    """
    Integrate the pendulum equations using the Euler-Cromer method.

    A first-order symplectic integrator that updates velocity before position,
    giving exact energy conservation on average. Superior to standard Euler
    for oscillatory systems despite being only first-order accurate.

    Parameters
    ----------
    theta0 : float
        Initial angular displacement (radians)
    omega0 : float
        Initial angular velocity (rad/s)
    t_final : float
        Total integration time (s)
    dt : float
        Fixed time step size (s)

    Returns
    -------
    t : ndarray
        Time array (s)
    theta : ndarray
        Angular displacement at each time step (radians)
    omega : ndarray
        Angular velocity at each time step (rad/s)
    """
    n_steps = int(t_final / dt)
    t = np.arange(n_steps) * dt
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0], omega[0] = theta0, omega0
    for i in range(n_steps - 1):
        alpha = angular_acceleration(theta[i])
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt
    return t, theta, omega


def solve_velocity_verlet(theta0, omega0, t_final, dt):
    """
    Integrate the pendulum equations using the Velocity Verlet method.

    A second-order symplectic integrator that evaluates acceleration at both
    the current and next positions, achieving better energy conservation and
    phase accuracy than Euler-Cromer with the same step size.

    Parameters
    ----------
    theta0 : float
        Initial angular displacement (radians)
    omega0 : float
        Initial angular velocity (rad/s)
    t_final : float
        Total integration time (s)
    dt : float
        Fixed time step size (s)

    Returns
    -------
    t : ndarray
        Time array (s)
    theta : ndarray
        Angular displacement at each time step (radians)
    omega : ndarray
        Angular velocity at each time step (rad/s)
    """
    n_steps = int(t_final / dt)
    t = np.arange(n_steps) * dt
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0], omega[0] = theta0, omega0
    for i in range(n_steps - 1):
        alpha = angular_acceleration(theta[i])
        theta[i + 1] = theta[i] + omega[i] * dt + 0.5 * alpha * dt**2
        alpha_new = angular_acceleration(theta[i + 1])
        omega[i + 1] = omega[i] + 0.5 * (alpha + alpha_new) * dt
    return t, theta, omega


def solve_yoshida4(theta0, omega0, t_final, dt):
    """
    Integrate the pendulum equations using the Yoshida 4th-order symplectic method.

    A fourth-order symplectic integrator constructed from three Verlet substeps
    with carefully chosen coefficients. Provides superior long-term energy
    conservation and spectral purity compared to lower-order symplectic methods.

    Parameters
    ----------
    theta0 : float
        Initial angular displacement (radians)
    omega0 : float
        Initial angular velocity (rad/s)
    t_final : float
        Total integration time (s)
    dt : float
        Fixed time step size (s)

    Returns
    -------
    t : ndarray
        Time array (s)
    theta : ndarray
        Angular displacement at each time step (radians)
    omega : ndarray
        Angular velocity at each time step (rad/s)
    """
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)
    c = np.array([w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0])
    d = np.array([w1, w0, w1])

    n_steps = int(t_final / dt)
    t = np.arange(n_steps) * dt
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0], omega[0] = theta0, omega0

    for i in range(n_steps - 1):
        th, om = theta[i], omega[i]
        for j in range(3):
            th += c[j] * om * dt
            om += d[j] * angular_acceleration(th) * dt
        th += c[3] * om * dt
        theta[i + 1], omega[i + 1] = th, om

    return t, theta, omega


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

    E0 = total_energy(theta_initial, omega_initial)

    # Expected fundamental (small-angle approximation for reference)
    f_natural = 1.0 / (2.0 * np.pi) * np.sqrt(G / LENGTH)

    methods = {
        "RK45 (solve_ivp)": (solve_rk45, "red"),
        "Euler-Cromer": (solve_euler_cromer, "orange"),
        "Velocity Verlet": (solve_velocity_verlet, "green"),
        "Yoshida 4th-Order": (solve_yoshida4, "blue"),
    }

    # Shared axis limits across all figures
    err_ylim = (1e-10, 1e0)
    db_min = -100
    db_max = 10

    results = {}
    for name, (solver, color) in methods.items():
        print(f"Running {name}...")
        t, theta, omega = solver(theta_initial, omega_initial, t_final, dt)
        E = total_energy(theta, omega)
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
