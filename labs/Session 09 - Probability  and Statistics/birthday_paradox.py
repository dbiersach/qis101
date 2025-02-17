#!/usr/bin/env python3
"""birthday_paradox.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

total_classes = 10_000
max_size = 80


@njit
def shared_birthdays(class_size):
    """Returns True/False if two students share a birthday"""
    b = np.random.randint(0, 365, class_size)
    for i in range(b.size - 2):
        for j in range(i + 1, b.size):
            if b[i] == b[j]:
                return True
    return False


@njit
def calc_probabilities():
    """Returns an array of probabilities of a shared birthday"""
    p = np.zeros(max_size + 1)
    for c in range(2, max_size + 1):
        n = 0
        for _ in range(total_classes):
            if shared_birthdays(c):
                n = n + 1
        p[c] = n / total_classes
    return p


def main():
    # Estimated probabilities of shared birthdays per class size
    prob = calc_probabilities()
    # Find the first class with shared birthday probability > 50%
    min_class_size = np.where(prob > 0.50)[0][0]

    # Actual probabilities of shared birthdays per class size
    x = np.linspace(0, 80, 500)
    y = 1.0 - np.exp(-(x**2) / 730)

    plt.figure(Path(__file__).name)
    plt.step(range(max_size + 1), prob, color="black", linewidth=3, label="Estimated")
    plt.plot(x, y, color="blue", label="Actual")
    plt.title(f"Birthday Paradox ({total_classes:,} classes)")
    plt.xlabel("Class Size (Number of Students)")
    plt.ylabel("Probability of a Shared Birthday")
    plt.vlines(min_class_size, 0, prob[min_class_size], color="red")
    plt.annotate(
        f"Min Class Size = {min_class_size}",
        xy=(min_class_size, float(prob[min_class_size])),
        xytext=(28, 0.45),
        arrowprops={"facecolor": "black"},
    )
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
