#!/usr/bin/env -S uv run
"""riemann_hypothesis.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def fn_eta(s):
    terms = 100_000
    evens = np.power(np.arange(2, terms, 2, dtype=complex), -s)
    odds = np.power(np.arange(3, terms, 2, dtype=complex), -s)
    return 1 - np.sum(evens) + np.sum(odds)


def fn_zeta_from_eta(s):
    return fn_eta(s) / (1.0 - np.power(2, 1.0 - s))


def main():
    xa = np.linspace(-1, 31, 1000)

    # Critical line: Re(s) = 0.5
    xz = [complex(0.5, i) for i in xa]

    print("Calculating Dirichlet eta function values...")
    eta = [fn_eta(s) for s in xz]

    print("Calculating Riemann zeta function values...")
    zeta = [fn_zeta_from_eta(s) for s in xz]
    zeta_zeros_im = [float(mp.im(mp.zetazero(n))) for n in range(1, 5)]

    plt.figure(Path(__file__).name)
    plt.plot(xa, np.absolute(eta), label=r"$\eta \left( s \right)$")
    plt.plot(xa, np.absolute(zeta), label=r"$\zeta \left( s \right)$", color="red")
    plt.scatter(
        zeta_zeros_im,
        np.zeros(len(zeta_zeros_im)),
        marker="o",
        color="green",
        label=r"$\zeta\ root$",
    )
    plt.title(r"Riemann $\zeta(s)$ vs. Dirichlet $\eta(s)$")
    plt.xlabel(r"$\mathrm{Im\left(s\right)}$")
    plt.ylabel(r"$\mathrm{|s|}$")
    ax = plt.gca()
    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed")
    plt.show()


if __name__ == "__main__":
    main()
