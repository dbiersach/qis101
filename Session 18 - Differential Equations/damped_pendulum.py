#!/usr/bin/env -S uv run
"""damped_pendulum.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import solve_ivp


def model(time, state_vector, phase_constant, damping_constant):
    omega, theta = state_vector  # unpack dependent variables
    d_omega = -damping_constant * omega - phase_constant * np.sin(theta)
    d_theta = omega
    return d_omega, d_theta


def main():
    # Precalculate phase constant
    pendulum_length = 1.0  # meters
    phase_constant = 9.81 / pendulum_length

    # Set damping_constants
    underdamped_constant = 1.0
    overdamped_constant = pow(phase_constant, 2) / 2.0
    critically_damped_constant = pow(phase_constant, 2) / 4.0

    # Set initial conditions
    omega_initial = 0
    theta_initial = np.radians(75)  # 75 degrees

    # Set model duration (seconds)
    time_initial = 0
    time_final = 15

    # Calculate for an underdamped pendulum
    sol = solve_ivp(
        model,
        (time_initial, time_final),
        [omega_initial, theta_initial],
        max_step=0.01,
        args=(phase_constant, underdamped_constant),
    )
    time_steps = sol.t
    theta_underdamped = sol.y[1]

    # Calculate for an overdamped pendulum
    sol = solve_ivp(
        model,
        (time_initial, time_final),
        [omega_initial, theta_initial],
        max_step=0.01,
        args=(phase_constant, overdamped_constant),
    )
    theta_overdamped = sol.y[1]

    # Calculate for a critically damped pendulum
    sol = solve_ivp(
        model,
        (time_initial, time_final),
        [omega_initial, theta_initial],
        max_step=0.01,
        args=(phase_constant, critically_damped_constant),
    )
    theta_critically_damped = sol.y[1]

    plt.figure(Path(__file__).name)
    # fmt: on
    plt.plot(time_steps, theta_underdamped, label="underdamped", c="r", lw=2, zorder=3)
    plt.plot(time_steps, theta_overdamped, label="overdamped", c="b", lw=2, zorder=3)
    plt.plot(
        time_steps,
        theta_critically_damped,
        label="critically damped",
        c="g",
        lw=2,
        zorder=3,
    )
    # fmt: on
    plt.title("Damped Pendulums")
    plt.xlabel("Time (sec)")
    plt.ylabel("Angular Displacement (radians)")
    plt.axhline(y=0.0, color="lightgray")
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", framealpha=1.0, facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
