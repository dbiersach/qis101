#!/usr/bin/env python3
"""estes_rocket.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

STAGE1_MASS = 0.0519  # kg
STAGE2_MASS = 0.1729  # kg

# For an Estes F15 rocket engine
ENGINE_MASS = 0.101  # kg
ENGINE_THRUST = 17.2  # Newtons
ENGINE_BURNOUT = 1.6  # seconds

G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M = 5.972e24  # Mass of the Earth (kg)
R = 6.371e6  # Radius of the Earth (m)


def thrust_func(t):
    # returns thrust in Newtons
    if t < ENGINE_BURNOUT * 2:
        return ENGINE_THRUST
    return 0


def rocket_mass_func(t):
    # returns weight in kilograms
    if t <= ENGINE_BURNOUT:
        motor_mass = ENGINE_MASS * (ENGINE_BURNOUT - t) / ENGINE_BURNOUT
        return STAGE1_MASS + motor_mass + STAGE2_MASS + ENGINE_MASS
    if t <= ENGINE_BURNOUT * 2:
        motor_mass = ENGINE_MASS * (ENGINE_BURNOUT * 2 - t) / ENGINE_BURNOUT
        return STAGE2_MASS + motor_mass
    return STAGE2_MASS


def model(t, state_vector):
    v, h = state_vector
    m = rocket_mass_func(t)
    F_thrust = thrust_func(t)
    F_gravity = G * M * m / (R + h) ** 2
    d_v = (F_thrust - F_gravity) / m
    d_h = v
    return d_v, d_h


def main():
    # Set simulation duration
    t0 = 0  # start time (seconds)
    tf = 4  # final time (seconds)

    # Set initial conditions
    y_0 = 0  # Initial height (m)
    v_0 = 0.0  # Initial velocity (m/s)

    sol = solve_ivp(model, (t0, tf), [v_0, y_0], max_step=0.01)
    t = sol.t
    v, h = sol.y

    v *= 2.23  # Convert m/s to mph
    h *= 3.28  # Convert m to feet

    plt.figure(Path(__file__).name, figsize=(10, 4))

    ax = plt.subplot(1, 2, 1)
    ax.plot(t, v, color="blue", lw=2)
    ax.set_title(f"Rocket Velocity - Mean: {np.mean(v):.2f} mph")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (mph)")
    ax.grid("on")

    ax = plt.subplot(1, 2, 2)
    ax.plot(t, h, color="orange", lw=2)
    ax.set_title(f"Rocket Altitude - Max: {np.max(h):,.2f} feet")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (feet)")
    ax.grid("on")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
