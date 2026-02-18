#!/usr/bin/env -S uv run
"""quadratic_regression.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def fit_quadratic(x, y):
    sum_x = sum(x)
    sum_x2 = sum(x**2)
    sum_x3 = sum(x**3)
    sum_x4 = sum(x**4)

    sum_y = sum(y)
    sum_xy = sum(x * y)
    sum_x2y = sum(x**2 * y)

    # fmt: off
    coeffs = np.array(
        [
            [sum_x4, sum_x3, sum_x2],
            [sum_x3, sum_x2, sum_x],
            [sum_x2, sum_x, len(x)]
        ]
    )
    # fmt: on

    vals = np.array([sum_x2y, sum_xy, sum_y])

    det_coeffs = np.linalg.det(coeffs)

    mat_a = np.copy(coeffs)
    mat_a[:, 0] = vals
    det_a = np.linalg.det(mat_a)

    mat_b = np.copy(coeffs)
    mat_b[:, 1] = vals
    det_b = np.linalg.det(mat_b)

    mat_c = np.copy(coeffs)
    mat_c[:, 2] = vals
    det_c = np.linalg.det(mat_c)

    a = det_a / det_coeffs
    b = det_b / det_coeffs
    c = det_c / det_coeffs

    return a, b, c


def fit_linear(x, y):
    m = len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)
    m = m / (len(x) * np.sum(x**2) - np.sum(x) ** 2)
    b = (np.sum(y) - m * np.sum(x)) / len(x)
    return m, b


def main():
    data_x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    data_y = np.array([0, 205, 430, 677, 945, 1233, 1542, 1872, 2224])

    plt.figure(Path(__file__).name, figsize=(10, 8))
    plt.scatter(data_x, data_y, c="k", zorder=2)

    x = np.linspace(np.min(data_x), np.max(data_x), 1000)

    m, b = fit_linear(data_x, data_y)
    plt.plot(x, m * x + b, label="Linear", c="b", zorder=2)

    a, b, c = fit_quadratic(data_x, data_y)
    plt.plot(x, a * x**2 + b * x + c, label="Quadratic", c="r")

    stop_time = 45
    stop_count = a * stop_time**2 + b * stop_time + c
    plt.scatter(stop_time, stop_count, marker="s", s=50, c="g", zorder=2)

    plt.title(
        (
            "Tape Counter Per Minute\n"  # noqa
            f"({stop_time} min = Counter {stop_count:,.0f})"
        )
    )
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel("Tape Counter")
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(MultipleLocator(100))
    plt.grid("on")
    plt.legend(loc="center right", framealpha=1.0, facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
