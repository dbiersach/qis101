#!/usr/bin/env -S uv run
"""ellipse_perimeter.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def d_s(theta, a, b):
    return np.sqrt(np.power(a * np.sin(theta), 2) + np.power(b * np.cos(theta), 2))


def ramanujan_estimate(a, b):
    h = (a - b) / (a + b)
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


def fit_quadratic(x, y):
    x = x[:, np.newaxis]
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    transformer.fit(x)
    x2 = np.array(transformer.transform(x))
    model = LinearRegression().fit(x2, y)
    a, b = model.coef_[1], model.coef_[0]
    c = model.intercept_
    return a, b, c


def plot_p(ax, p, r):
    ax.plot(range(len(p)), p, label="Integral")
    ax.plot(range(len(r)), r, label="Ramanujan")
    ax.set_title("Numerical Ellipse Perimeter Estimate")
    ax.set_xlabel("b")
    ax.set_ylabel("Perimeter")
    ax.legend(loc="upper left")
    ax.set_xlim(1, len(p) - 1)


def plot_err(ax, err):
    ax.scatter(range(len(err)), err, color="red")
    ax.set_title("Ramanujan's Estimate Relative Error")
    ax.set_xlabel("b")
    ax.set_ylabel("Relative Error")
    ax.set_xlim(1, len(err) - 1)


def plot_fit(ax, err, fa, fb, fc):
    ax.scatter(range(len(err)), err, color="red")
    x = np.linspace(0, len(err) - 1, 500)
    ax.plot(x, fa * x**2 + fb * x + fc)
    ax.set_title("Ramanujan's Error (Quadratic Fit)")
    ax.set_xlabel("b")
    ax.set_ylabel("Relative Error")
    ax.set_xlim(1, len(err) - 1)


def plot_fix(ax, p, f):
    ax.plot(range(len(p)), p, label="Integral")
    ax.plot(range(len(f)), f, label="Adjusted")
    ax.set_title("Ramanujan's Perimeter Estimate (Adjusted)")
    ax.set_xlabel("b")
    ax.set_ylabel("Perimeter")
    ax.legend(loc="upper left")
    ax.set_xlim(1, len(f) - 1)


def main():
    a, b_max = 100, 21
    peri, ram, err = (np.zeros(b_max) for _ in range(3))

    for b in range(0, b_max):
        peri[b] = quad(d_s, 0, 2 * np.pi, args=(a, b))[0]
        ram[b] = ramanujan_estimate(a, b)
        err[b] = (ram[b] - peri[b]) / ram[b]

    fit_a, fit_b, fit_c = fit_quadratic(np.arange(len(err)), err)
    print("Quadratic Error Adjuster:")
    print(f"{fit_a:.5f}x^2 + {fit_b:.5f}x + {fit_c:.5f}")

    adj = np.zeros(b_max)
    print(f"{'b':>3}{'Perimeter':>10}{'Ramanujan':>11}{'Error':>10}{'Adjusted':>10}")
    for b in range(b_max):
        adj[b] = ram[b] - ram[b] * (fit_a * b**2 + fit_b * b + fit_c)
        print(f"{b:>3}{peri[b]:>10.3f}{ram[b]:>11.3f}{err[b]:>10.6f}{adj[b]:>10.3f}")

    plt.figure(Path(__file__).name, figsize=(12, 8))
    plot_p(plt.subplot(2, 2, 1), peri, ram)
    plot_err(plt.subplot(2, 2, 2), err)
    plot_fit(plt.subplot(2, 2, 3), err, fit_a, fit_b, fit_c)
    plot_fix(plt.subplot(2, 2, 4), peri, adj)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
