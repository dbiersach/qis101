#!/usr/bin/env -S uv run
"""quadratic_regression_sklearn.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def fit_linear(x, y):
    # Reshape vector x to become matrix x
    x = x[:, np.newaxis]
    model = LinearRegression().fit(x, y)
    m = model.coef_[0]
    b = model.intercept_
    return m, b


def fit_quadratic(x, y):
    # Reshape vector x to become matrix x
    x = x[:, np.newaxis]
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    transformer.fit(x)
    # The matrix x2 will have two columns:
    # 1) the original x values and 2) the x**2 values
    x2 = np.array(transformer.transform(x))
    model = LinearRegression().fit(x2, y)
    a = model.coef_[1]
    b = model.coef_[0]
    c = model.intercept_
    return a, b, c


def main():
    data_x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    data_y = np.array([0, 205, 430, 677, 945, 1233, 1542, 1872, 2224])

    plt.figure(Path(__file__).name, figsize=(10, 8))
    plt.scatter(data_x, data_y, color="k", zorder=2)

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
