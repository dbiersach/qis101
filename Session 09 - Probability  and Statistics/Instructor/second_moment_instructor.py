#!/usr/bin/env python3
"""second_moment_instructor.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def main():
    dice = [4, 6, 8, 10, 12, 20]  # number of sides of die (6 total dice)
    magic = np.zeros((6, 21))  # Magic number 2D matrix [6 rows x 21 columns]
    for idx, sides in enumerate(dice):  # for each die...
        for n in range(21):  # 0 <= n < 21 (n = col number in 2D magic matrix)
            count = 2 ** (n + 4)  # actual total count of rolls to make
            rolls = np.random.randint(1, sides + 1, count)  # array of random rolls
            var = np.var(rolls)  # population variance of random rolls
            magic[idx, n] = (sides**2 - 1) / var  # calculate magic number

    # Print the final (largest roll count) magic number for each die
    for idx, sides in enumerate(dice):
        print(f"Sides = {sides:3}: Magic Number = {magic[idx, -1]:>8.5f}")

    # Plot the magic number as a function of die sides and count of rolls
    plt.figure(Path(__file__).name)
    for idx, sides in enumerate(dice):
        plt.plot(np.arange(21) + 4, magic[idx, :], label=f"{sides} sides")

    plt.title("Magic Number (per die sides)")
    plt.xlabel("Number of rolls $(2^x)$")
    plt.ylabel("Magic Number")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
