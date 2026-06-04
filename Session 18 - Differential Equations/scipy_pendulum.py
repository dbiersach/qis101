#!/usr/bin/env -S uv run
"""scipy_pendulum.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def model(time, state_vector, phase_constant):
    # Unpack current state vector (dependent variables)
    omega, theta = state_vector
    # Express differential equations
    d_omega = -phase_constant * np.sin(theta)
    d_theta = omega
    return d_omega, d_theta


def main():
    # Precalculate phase constant
    pendulum_length = 1.0  # meters
    phase_constant = 9.81 / pendulum_length

    # Set initial conditions
    omega_initial = 0
    theta_initial = np.deg2rad(45)

    # Set model duration (seconds)
    time_initial = 0
    time_final = 10

    # Numerically estimate the ODE using RK45 Method
    sol = solve_ivp(
        model,
        (time_initial, time_final),  # tuple of time span
        [omega_initial, theta_initial],  # initial state vector
        max_step=0.01,
        args=(phase_constant,),  # tuple of constants used in ODE
    )
    # Retrieve results of the solution
    time_steps = sol.t
    omega, theta = sol.y  # Unpack order like initial state vector

    _, ax_theta = plt.subplots(num=Path(__file__).name)
    (plot1,) = ax_theta.plot(
        time_steps, theta, lw=2, label=r"$\theta$ (rad)", color="C0"
    )
    ax_theta.set_title("Simple Pendulum (RKF45 Method)")
    ax_theta.set_xlabel("Time (sec)")
    ax_theta.set_ylabel(r"Angular Displacement $\theta$ (rad)", color="C0")
    ax_theta.tick_params(axis="y", labelcolor="C0")
    ax_theta.grid(True)
    ax_omega = ax_theta.twinx()
    (plot2,) = ax_omega.plot(
        time_steps, omega, lw=2, label=r"$\omega$ (rad/s)", color="C1"
    )
    ax_omega.set_ylabel(r"Angular Velocity $\omega$ (rad/s)", color="C1")
    ax_omega.tick_params(axis="y", labelcolor="C1")
    # fmt: off
    ax_theta.legend([plot1, plot2], [r"$\theta$", r"$\omega$"],
        loc="upper right", framealpha=1.0, facecolor="white")
    # fmt: on
    plt.show()


if __name__ == "__main__":
    main()
