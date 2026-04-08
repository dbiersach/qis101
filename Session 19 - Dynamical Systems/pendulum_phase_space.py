#!/usr/bin/env -S uv run
"""pendulum_phase_space.py

Plots the phase space (theta vs omega) for an ideal pendulum,
comparing four numerical integrators against the exact energy
contour.

  Black dashed: Exact energy contour E(theta, omega) = E0
  Red:          Forward Euler    - spirals outward (energy gain)
  Orange:       Euler-Cromer     - wobbles around the contour
  Green:        Velocity Verlet  - hugs the contour tightly
  Blue:         Yoshida 4th-ord  - indistinguishable from exact

The exact contour is computed from energy conservation:
  E = (1/2)omega^2 - (g/L)cos(theta)
so for a given E0, omega = +/- sqrt(2(E0 + (g/L)cos(theta)))

Yoshida coefficients (Forest-Ruth / Yoshida 4th-order):
  w1 = 1 / (2 - 2^(1/3))
  w0 = -2^(1/3) * w1
  c  = [w1/2, (w1+w0)/2, (w0+w1)/2, w1/2]   (position sub-steps)
  d  = [w1, w0, w1]                            (momentum sub-steps)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qis101_utils import (
    PENDULUM_G,
    PENDULUM_LENGTH,
    pendulum_angular_acceleration,
    pendulum_euler_cromer,
    pendulum_total_energy,
    pendulum_velocity_verlet,
    pendulum_yoshida4,
)

LINE_WIDTH = 2  # uniform linewidth for all plots


# ========================================================================
# Integrators (all return t, theta, omega arrays)
# ========================================================================


def solve_forward_euler(theta0, omega0, t_final, dt):
    """
    Integrate the pendulum equations using the Forward Euler method.

    A first-order non-symplectic integrator that updates position using the
    current velocity before updating velocity. Energy grows monotonically,
    causing the phase-space trajectory to spiral outward over time.

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
        alpha = pendulum_angular_acceleration(theta[i])
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i] * dt
    return t, theta, omega


# ========================================================================
# Exact energy contour
# ========================================================================


def exact_contour(E0, n_points=1000):
    """
    Compute the exact closed phase-space contour for a given energy E0.

    Derives the turning angles analytically from energy conservation, then
    samples theta symmetrically so both tips land exactly at ω=0 with no gap.
    Stitches the upper and lower half-orbits into one closed loop.

    Parameters
    ----------
    E0 : float
        Total mechanical energy per unit mass (J/kg)
    n_points : int, optional
        Number of points along each half of the contour; default is 1000

    Returns
    -------
    theta_closed : ndarray
        Angular displacement values forming the closed contour (radians)
    omega_closed : ndarray
        Angular velocity values forming the closed contour (rad/s)
    """
    # Exact turning angle: omega=0 when E0 + (g/L)cos(theta) = 0
    theta_max = np.arccos(-E0 * PENDULUM_LENGTH / PENDULUM_G)

    theta_valid = np.linspace(-theta_max, theta_max, n_points)
    omega_sq = 2.0 * (E0 + (PENDULUM_G / PENDULUM_LENGTH) * np.cos(theta_valid))
    omega_upper = np.sqrt(np.maximum(omega_sq, 0.0))
    omega_lower = -omega_upper

    # Stitch into one closed loop: forward along top, backward along bottom
    theta_closed = np.concatenate([theta_valid, theta_valid[::-1]])
    omega_closed = np.concatenate([omega_upper, omega_lower[::-1]])

    return theta_closed, omega_closed


def main():
    # --- Simulation parameters ---
    theta_initial = np.deg2rad(45)
    omega_initial = 0.0
    dt = 0.05
    t_symplectic = 500  # seconds for symplectic methods
    t_euler = 20  # shorter for Forward Euler (it blows up fast)

    E0 = pendulum_total_energy(theta_initial, omega_initial)

    # --- Run integrators ---
    methods = {
        "Forward Euler": (solve_forward_euler, "purple", 0.7, t_euler),
        "Euler-Cromer": (pendulum_euler_cromer, "orange", 0.7, t_symplectic),
        "Velocity Verlet": (pendulum_velocity_verlet, "green", 0.9, t_symplectic),
        "Yoshida 4th-order": (pendulum_yoshida4, "blue", 0.9, t_symplectic),
    }

    results = {}
    for name, (solver, color, alpha, t_run) in methods.items():
        print(f"Running {name} for {t_run}s...")
        t, theta, omega = solver(theta_initial, omega_initial, t_run, dt)
        results[name] = (t, theta, omega, color, alpha)

    # --- Exact contour (closed loop, no gap at turning points) ---
    theta_exact, omega_exact = exact_contour(E0)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.canvas.manager.set_window_title(f"{Path(__file__).name} - Phase Space")

    # Numerical trajectories first so the exact contour is drawn on top
    for name in methods:
        t, theta, omega, color, alpha = results[name]
        ax.plot(
            theta, omega, color=color, linewidth=LINE_WIDTH, alpha=alpha, label=name
        )

    # Exact contour drawn last - always visible above numerical curves
    ax.plot(theta_exact, omega_exact, "r--", lw=2, label="Exact contour")

    # Clip axes to focus on the contour region
    pad = 0.3
    ax.set_xlim(theta_exact.min() - pad, theta_exact.max() + pad)
    ax.set_ylim(omega_exact.min() - 1.5, omega_exact.max() + 1.5)

    ax.set_title(
        f"Phase Space [dt={dt}s, "
        rf"$\theta_0$={np.rad2deg(theta_initial):.0f}°]"
        f"   (Euler: {t_euler}s, others: {t_symplectic}s)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\omega$ (rad/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # --- Print energy drift summary ---
    print(f"\nE0 = {E0:.6f}")
    print(
        f"{'Method':<22} {'t_run (s)':>10} {'Final E':>12} {'Drift':>12} {'Drift %':>10}"
    )
    print("-" * 70)
    for name in methods:
        t, theta, omega, color, alpha = results[name]
        E_final = pendulum_total_energy(theta[-1], omega[-1])
        drift = E_final - E0
        drift_pct = 100 * drift / abs(E0)
        print(
            f"{name:<22} {t[-1]:>10.0f} {E_final:>12.6f} {drift:>+12.6f} {drift_pct:>+10.4f}%"
        )

    plt.show()


if __name__ == "__main__":
    main()
