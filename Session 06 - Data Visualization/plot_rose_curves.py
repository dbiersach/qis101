#!/usr/bin/env -S uv run

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes  # for type hinting
from numpy import ndarray  # for type hinting

"""
Technically these are not true "rose curves" since those have equations of the form
r = a * cos(k*theta) or r = a * sin(k*theta). Here we have added a constant term to
create more interesting shapes. These are sometimes called limaçon ("snail") curves.
For more information, see:
https://mathworld.wolfram.com/Limacon.html 
https://en.wikipedia.org/wiki/Lima%C3%A7on
"""


def r1(theta: ndarray) -> ndarray:
    return 4 + 4 * np.cos(4 * theta)


def r2(theta: ndarray) -> ndarray:
    return 3 + 3 * np.cos(4 * theta + np.pi)


def r3(theta: ndarray) -> ndarray:
    return 5 + 5 * np.cos(3 / 2 * theta)


def plot(ax: Axes) -> None:
    theta = np.linspace(0, 4 * np.pi, 1000)
    ax.plot(theta, r1(theta), label=r"$4+4\cos{4\theta}$")
    # ax.plot(theta, r2(theta), label=r"$3+3\cos{(4\theta+\pi)}$")
    # ax.plot(theta, r3(theta), label=r"$5+5\cos{(\dfrac{3}{2}\theta)}$")
    ax.set_title("Rose Curves")
    ax.legend(loc="upper right")


def main() -> None:
    plt.figure(Path(__file__).name, figsize=(8, 6))
    ax: Axes = plt.axes(projection="polar")
    plot(ax)
    plt.show()


if __name__ == "__main__":
    main()
