#!/usr/bin/env -S uv run
"""taylor_series.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy


def main():
    # Plot exact y = cos(x)
    x = np.linspace(0, 2 * np.pi, 1000)
    plt.figure(Path(__file__).name)
    plt.plot(x, np.cos(x), label="Exact")

    # Plot Taylor Series for cos(x)
    x_taylor = sympy.symbols("x")
    poly = sympy.cos(x_taylor).series(x_taylor, 0, 9).removeO()
    eqn = sympy.lambdify(x_taylor, poly.as_expr(), modules="numpy")
    print(f"Taylor Series for cos(x) = {poly}")
    num_terms_taylor = len(poly.as_ordered_terms())
    plt.plot(x, eqn(x), label=f"Taylor ({num_terms_taylor} terms)")

    # Plot Euler's Method for d[cos(x)] = -sin(x)
    num_terms_euler = 20
    x_euler = np.linspace(0, 2 * np.pi, num_terms_euler)
    dx_euler = x_euler[1] - x_euler[0]
    y_euler = np.zeros(num_terms_euler)
    y_euler[0] = np.cos(x_euler[0])
    for i in range(1, len(x_euler)):
        y_euler[i] = y_euler[i - 1] - np.sin(x_euler[i - 1]) * dx_euler
    plt.plot(x_euler, y_euler, label=f"Euler ({num_terms_euler} intervals)")

    plt.title(r"Series Comparison for $y = \cos(x)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-1.1, 1.1)
    plt.axhline(y=0.0, color="lightgray", zorder=-2)
    plt.legend(loc="lower left")
    plt.show()


if __name__ == "__main__":
    main()
