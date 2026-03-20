#!/usr/bin/env -S uv run
"""bessel_correction.py"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from numba import njit
from numpy.random import choice, randint


@njit
def get_bsv(arr):
    mean = np.mean(arr)
    bsv = float(np.sum((arr - mean) ** 2) / len(arr))
    return bsv


@njit
def get_sample_bsv(population, sample_size):
    num_trials = 20_000
    total_bsv = 0.0
    for _ in range(num_trials):
        samples = choice(population, sample_size, replace=False)
        total_bsv += get_bsv(samples)
    mean_bsv = total_bsv / num_trials
    return mean_bsv


def run_trials():
    population = randint(0, 1000, 7000)
    pop_var = get_bsv(population)

    max_sample_size = 20

    print(f"{'Sample Size':^11}{'Sample Var':^21}{'Pop Var':^18}{'Ratio':^8}")

    results = []
    for sample_size in range(2, max_sample_size + 1):
        sample_bsv = get_sample_bsv(population, sample_size)
        ratio = sample_bsv / pop_var
        results.append((sample_size, sample_bsv, pop_var, ratio))
        print(
            f"{sample_size:^11}{sample_bsv:>16,.4f}{pop_var:>18,.4f}", f"{ratio:^15.4f}"
        )
    return results


def plot_ratio(ax, results):
    x1 = [r[0] for r in results]  # 1st column in results table
    y1 = [r[3] for r in results]  # 4th column in results table
    ax.plot(x1, y1, label="BSV/PV")

    x2 = np.linspace(2, 20)
    y2 = (x2 - 1) / x2
    ax.plot(x2, y2, label=r"$\frac{n-1}{n}$")

    ax.set_title(r"$\frac{BSV}{PV}$ compared to Hyperbola $\frac{(n-1)}{n}$")
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Biased Sample Var / Population Var")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.legend(loc="center right")


def plot_ubsv(ax, results):
    x = [r[0] for r in results]  # 1st column in results table
    y = [r[2] for r in results]  # 3rd column in results table
    ax.plot(x, y, label="Pop Var")

    y = [r[1] for r in results]
    ax.plot(x, y, label="BSV")

    for i, _ in enumerate(y):
        y[i] = y[i] * x[i] / (x[i] - 1)
    ax.plot(x, y, label="UBSV")

    ax.set_title("Variance: Population v. Biased Sample v. Unbiased Sample")
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Variance")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="center right")


def main():
    plt.figure(Path(__file__).name)
    file_name = "bessel.pickle"
    data_file = Path(__file__).parent / file_name
    if not data_file.exists():
        results = run_trials()
        with open(data_file, "wb") as file_out:
            pickle.dump(results, file_out, pickle.HIGHEST_PROTOCOL)
        plot_ratio(plt.gca(), results)
    else:
        with open(data_file, "rb") as file_in:
            results = pickle.load(file_in)
        print(f"{'Sample Size':^11}{'Sample Var':^21}{'Pop Var':^16}{'UBSV':^12}")
        for r in results:
            sample_size, sample_bsv, pop_var, _ = r
            ubsv = sample_bsv * sample_size / (sample_size - 1)
            print(
                f"{sample_size:^11}{sample_bsv:>16,.4f}{pop_var:>18,.4f}",
                f"{ubsv:^18,.4f}",
            )
        plot_ubsv(plt.gca(), results)

    plt.show()


if __name__ == "__main__":
    main()
