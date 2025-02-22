#!/usr/bin/env python3
"""harmonograph.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import solve_ivp


def model(time, state_vector, phase_constant):
    omega, theta = state_vector
    d_omega = -phase_constant * np.sin(theta)
    d_theta = omega
    return d_omega, d_theta


def main():
    # Set Initial Conditions
    pendulum1_length = 1.0  # meters
    theta1_initial = np.radians(10)  # degrees
    omega1_initial = 0.0  # radians/sec

    pendulum2_length = 1.5  # meters
    theta2_initial = np.radians(10)  # degrees
    omega2_initial = 0.0  # radians/sec

    # Precalculate phase constants
    phase1_constant = 9.81 / pendulum1_length
    phase2_constant = 9.81 / pendulum2_length

    # Set model duration (seconds)
    time_initial = 0
    time_final = 10

    # Calculate trajectory of 1st pendulum
    sol = solve_ivp(
        model,
        (time_initial, time_final),
        [omega1_initial, theta1_initial],
        max_step=0.01,
        args=(phase1_constant,),
    )
    theta1 = sol.y[1]

    # Calculate trajectory of 2nd pendulum
    sol = solve_ivp(
        model,
        (time_initial, time_final),
        [omega2_initial, theta2_initial],
        max_step=0.01,
        args=(phase2_constant,),
    )
    theta2 = sol.y[1]

    # Ensure both vectors are same length
    if len(theta1) > len(theta2):
        theta1 = theta1[: len(theta2)]
    else:
        theta2 = theta2[: len(theta1)]

    plt.figure(Path(__file__).name)
    plt.plot(theta1, theta2, color="blue", lw=2)
    plt.title("Two Pendulum Harmonograph")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.show()


if __name__ == "__main__":
    main()
