#!/usr/bin/env -S uv run
"""pendulum_yoshida.py

Implements the Yoshida 4th-order symplectic integrator to simulate a simple pendulum.
This composes three weighted leapfrog steps per time step to cancel lower-order errors
From Haruo Yoshida's 1990 paper "Construction of Higher Order Symplectic Integrators"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qis101_utils import pendulum_yoshida4


def main():
    # Simulation parameters
    theta0 = np.deg2rad(45)  # initial angle (radians)
    omega0 = 0.0  # initial angular velocity (rad/s)
    tf = 10  # final time (s)
    dt = tf / 500  # time step - 500 steps, updates every 20 ms

    mass = 1.0  # pendulum bob mass (kg)
    length = 1.0  # pendulum length (m)
    g = 9.81  # gravitational acceleration (m/s^2)

    t, theta, omega = pendulum_yoshida4(theta0, omega0, tf, dt)

    # Initial and final energies
    Ei = 0.5 * mass * length**2 * omega[0] ** 2
    Ei += mass * g * length * (1 - np.cos(theta[0]))
    Ef = 0.5 * mass * length**2 * omega[-1] ** 2
    Ef += mass * g * length * (1 - np.cos(theta[-1]))
    # Energy drift in joules
    drift_pct = 100 * (Ef - Ei) / Ei
    print(f"Initial energy : {Ei:6.3f} J")
    print(f"Final energy   : {Ef:6.3f} J")
    print(f"Energy drift   : {Ef - Ei:6.3f} J  ({drift_pct:6.3f}%)")

    fig, ax1 = plt.subplots(num=Path(__file__).name, figsize=(9, 5))

    (plot1,) = ax1.plot(t, theta, lw=2, label=r"$\theta$ (rad)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"Angular Displacement $\theta$ (rad)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    (plot2,) = ax2.plot(t, omega, lw=2, color="orange", label=r"$\omega$ (rad/s)")
    ax2.set_ylabel(r"Angular Velocity $\omega$ (rad/s)")

    plt.title("Simple Pendulum (Yoshida 4th-Order Symplectic)")
    plt.legend(
        [plot1, plot2], [r"$\theta$", r"$\omega$"], framealpha=1.0, facecolor="white"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
