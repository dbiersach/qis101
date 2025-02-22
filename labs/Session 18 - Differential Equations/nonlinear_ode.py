#!/usr/bin/env python3
"""nonlinear_ode.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def dy(x, y):
    return y * np.cos(x)


def main():
    xf = 12 * np.pi  # final x value
    xs = 1000  # total steps across x-axis from origin
    dx = xf / xs  # step size across x-axis from origin

    x = np.zeros(xs)
    y1 = np.zeros(xs)
    y2 = np.zeros(xs)

    # Initial condition
    y1[0] = 1
    y2[0] = 1

    # Numerically estimate ODE using Euler and RK4 Methods
    for i in range(xs - 1):
        x[i + 1] = x[i] + dx

        # Euler's Method
        y1[i + 1] = y1[i] + dy(x[i], y1[i]) * dx

        # 4th Order Runge-Kutta Method
        k1 = dy(x[i], y2[i])
        k2 = dy(x[i], y2[i] + k1 / 2 * dx)
        k3 = dy(x[i], y2[i] + k2 / 2 * dx)
        k4 = dy(x[i], y2[i] + k3 * dx)
        y2[i + 1] = y2[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dx

    plt.figure(Path(__file__).name)
    plt.plot(x, y1, lw=2, label="Euler")
    plt.plot(x, y2, lw=2, label=r"Runge-Kutta $\mathcal{O}(h^4)$")
    # fmt: off
    plt.title(
        r"$\frac{dy}{dx} = y\cdot\cos(x),\; y(0)=1"
        r"\quad\rightarrow\quad y=e^{\sin(x)}$"
    )
    # fmt: on
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
