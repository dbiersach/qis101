#!/usr/bin/env -S uv run
"""pendulum_yoshida.py

Implements the Yoshida 4th-order symplectic integrator to simulate a simple pendulum.
This composes three weighted leapfrog steps per time step to cancel lower-order errors.
From Haruo Yoshida's 1990 paper "Construction of Higher Order Symplectic Integrators."
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

    t, theta, omega = pendulum_yoshida4(theta0, omega0, tf, dt)

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
