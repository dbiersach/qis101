#!/usr/bin/env -S uv run
"""uncertainty_principle.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def f(x):
    # The Gaussian standard normal probability density function
    return (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp((-1 / 2) * np.power(x, 2) / (np.power(sigma, 2)))
    )


def plot_samples(ts, ys, ax):
    global wave_pdf
    (wave_pdf,) = ax.plot(ts, ys, animated=True)
    ax.set_title("Particle Location (Probability Density)")
    ax.set_xlabel("Location", loc="right")
    ax.set_ylabel("Probability Density")
    ax.set_ylim(0, 25)


def plot_powerspec(ps, ax):
    global wave_ps
    (wave_ps,) = ax.plot(range(len(ps)), ps, color="green", animated=True)
    ax.set_title("Particle Frequencies")
    ax.set_xlabel("Frequency", loc="right")
    ax.set_ylabel(r"$\Vert Amplitude\Vert$")
    ax.set_ylim(0, 1)


def anim_frame_counter():
    global sigma
    n = 1
    while n < 1200:
        sigma = 10 / n if n <= 600 else 10 / (1200 - n)
        n += 1
        yield n


def anim_draw_frame(t):
    global sigma
    ys = f(ts)
    wave_pdf.set_data(ts, ys)

    sigma /= 30
    ys = f(ts)
    ca = np.fft.rfft(ys) / 2
    ps = np.abs(ca) / len(ca)

    wave_ps.set_data(range(len(ps)), ps)

    return (
        wave_pdf,
        wave_ps,
    )


def main():
    global ts, sigma, anim

    ts = np.linspace(-1, 1, 1000, endpoint=False)

    sigma = 1
    ys = f(ts)

    ca = np.fft.rfft(ys) / 2
    ps = np.abs(ca) / len(ca)

    plt.figure(Path(__file__).name, figsize=(12, 4))

    ax_pdf = plt.subplot(1, 2, 1)
    ax_ps = plt.subplot(1, 2, 2)

    plot_samples(ts, ys, ax_pdf)
    plot_powerspec(ps, ax_ps)

    # fmt: off
    anim = FuncAnimation(ax_pdf.figure,
        anim_draw_frame, anim_frame_counter, interval=25,
        cache_frame_data=False, blit=True, repeat=False)
    # fmt: on

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
