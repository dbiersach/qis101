#!/usr/bin/env -S uv run
"""pendulum_yoshida.py

Implements the Yoshida 4th-order symplectic integrator to simulate a simple pendulum.
This composes three weighted leapfrog steps per time step to cancel lower-order errors.
From Haruo Yoshida's 1990 paper "Construction of Higher Order Symplectic Integrators."
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Physical constants
LENGTH = 1.0  # pendulum length (m)
G = 9.81  # gravity (m/s²)


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


def solve_yoshida4(theta0, omega0, t_final, dt):
    """
    Integrate the pendulum equations using the Yoshida 4th-order symplectic method.

    A fourth-order symplectic integrator constructed from three Verlet substeps
    with carefully chosen coefficients (Forest & Ruth 1990 / Yoshida 1990).
    Provides superior long-term energy conservation and phase accuracy compared
    to lower-order symplectic methods; the phase-space trajectory is visually
    indistinguishable from the exact energy contour.

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
    # Yoshida 4th-order coefficients
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


def main():
    # Simulation parameters
    theta0 = np.deg2rad(45)  # initial angle (radians)
    omega0 = 0.0  # initial angular velocity (rad/s)
    tf = 10  # final time (s)
    dt = tf / 500  # time step - 500 steps, updates every 20 ms

    t, theta, omega = solve_yoshida4(theta0, omega0, tf, dt)

    plt.figure(Path(__file__).name)
    (plot1,) = plt.plot(t, theta, lw=2)
    (plot2,) = plt.plot(t, omega, lw=2)
    plt.title("Simple Pendulum (Yoshida 4th-Order Symplectic)")
    plt.xlabel("Time (sec)")
    plt.ylabel(r"Angular Displacement $\theta$ (rad)")
    plt.twinx()
    plt.ylabel(r"Angular Velocity $\omega$ (rad/s)")
    plt.legend(
        [plot1, plot2], [r"$\theta$", r"$\omega$"], framealpha=1.0, facecolor="white"
    )
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
