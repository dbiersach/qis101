#!/usr/bin/env -S uv run
"""irregular_integration.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson


def f(theta):
    return np.sin(theta) ** 2


def F(theta):
    return (theta - np.sin(theta) * np.cos(theta)) / 2


def main():
    a, b, n = 0, np.pi, 50  # n = Number of samples

    # Generate 'n' sorted random angles (in radians)
    np.random.seed(2019)
    theta_samples = np.sort((b - a) * np.random.rand(n))
    # Samples the curve at those angles
    f_samples = f(theta_samples)

    # Compute the integral of the curve using the random
    # sample values and the *cumulative* Simpson's rule
    area_discrete = cumulative_simpson(f_samples, x=theta_samples)

    # Compare the discrete vs. analytic integral value
    print(f"Discrete Integral: {area_discrete[-1]:0.4f}")
    print(f"Analytic Integral: {F(b) - F(a):0.4f}")

    # Plot discrete (sampled) data
    plt.figure(Path(__file__).name)
    theta = np.linspace(0, np.pi, 1000)
    plt.plot(theta, f(theta), label=r"$\sin^2\theta$")
    plt.scatter(theta_samples, f_samples, c="r", label=f"samples ({n})")
    plt.title("Integration over Irregularly Spaced Points")
    plt.xlabel(r"$\theta$ (radians)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
