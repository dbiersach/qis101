#!/usr/bin/env -S uv run
"""pendulum_verlet.py

Implements the Velocity Verlet method to simulate a simple pendulum.
This is a 2nd-order symplectic integrator suitable for Hamiltonian systems.
From Loup Verlet's 1967 paper "Computer Experiments on Classical Fluids."
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


def main():
    # Simulation parameters
    theta0 = np.deg2rad(45)  # initial angle (radians)
    omega0 = 0.0  # initial angular velocity (rad/s)
    tf = 10  # final time (s)
    dt = tf / 500  # time step - 500 steps, updates every 20 ms

    t, theta, omega = solve_velocity_verlet(theta0, omega0, tf, dt)

    plt.figure(Path(__file__).name)
    (plot1,) = plt.plot(t, theta, lw=2)
    (plot2,) = plt.plot(t, omega, lw=2)
    plt.title("Simple Pendulum (Velocity Verlet Method)")
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
