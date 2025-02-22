#!/usr/bin/env python3
"""epidemiology.py"""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def model(time, state_vector, beta, delta):
    s, i, r = state_vector
    d_s = -beta * s * i
    d_i = beta * s * i - delta * i
    d_r = delta * i
    return d_s, d_i, d_r


def main():
    # Set model duration
    t_0, t_f = 0, 10  # months

    # Set simulation constants
    beta = 0.003  # Infection rate
    delta = 1  # Recovery rate

    # Set initial conditions
    s_0 = 1000  # Initial tally of susceptible people
    i_0 = 1  # Initial tally of infected people
    r_0 = 0  # Initial tally of recovered people

    # fmt: off
    sol = solve_ivp(model, (t_0, t_f), [s_0, i_0, r_0],
        max_step=0.01, args=(beta, delta))
    # fmt: on

    # Retrieve results of the solution
    t = sol.t
    s, i, r = sol.y

    # Plot S-I-R compartment populations over time
    plt.figure(Path(__file__).name)
    plt.plot(t, s, lw=2, label="Susceptible")
    plt.plot(t, i, lw=2, label="Infected")
    plt.plot(t, r, lw=2, label="Recovered")
    plt.title("S-I-R Epidemiology (Kermack-McKendrick)")
    plt.xlabel("Time (months)")
    plt.ylabel("Population")
    plt.legend(loc="center right", framealpha=1.0, facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
