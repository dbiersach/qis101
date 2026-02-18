#!/usr/bin/env -S uv run
"""ladder_problem.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from matplotlib.ticker import AutoMinorLocator


def length(theta):
    return 2 / np.sin(theta) + 1 / np.cos(theta)


def main():
    # Don't include the start or end points as they cause division by zero
    # in either the first or second term of the sum for length
    theta = np.linspace(0, np.pi / 2, 100)[1:-2]
    ladder = length(theta)

    # Find point where ladder length is CONSTANT
    # This means the first derivative of length() is zero at an extrema point
    # In this case that extrema is a global minimum of length()
    result = scipy.optimize.minimize(length, np.pi / 4)

    # The result.x is an array, and we only want the first element
    c_theta = result.x[0]
    c_length = length(c_theta)

    plt.figure(Path(__file__).name)
    plt.plot(theta, ladder)
    plt.plot(c_theta, c_length, color="green", marker="o")
    plt.title(f"Max Length = {c_length:.4f} m")
    plt.xlabel(r"$\theta$ (radians)")
    plt.ylabel("Length (m)")
    plt.ylim(0, 25)
    plt.axhline(c_length, color="gray", linestyle="--")
    plt.vlines(x=c_theta, ymin=0, ymax=c_length, color="red")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.show()


if __name__ == "__main__":
    main()
