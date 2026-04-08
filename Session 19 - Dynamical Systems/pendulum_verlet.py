#!/usr/bin/env -S uv run
"""pendulum_verlet.py

Implements the Velocity Verlet method to simulate a simple pendulum.
This is a 2nd-order symplectic integrator suitable for Hamiltonian systems.
From Loup Verlet's 1967 paper "Computer Experiments on Classical Fluids."
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qis101_utils import pendulum_velocity_verlet


def main():
    # Simulation parameters
    theta0 = np.deg2rad(45)  # initial angle (radians)
    omega0 = 0.0  # initial angular velocity (rad/s)
    tf = 10  # final time (s)
    dt = tf / 500  # time step - 500 steps, updates every 20 ms

    t, theta, omega = pendulum_velocity_verlet(theta0, omega0, tf, dt)

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
