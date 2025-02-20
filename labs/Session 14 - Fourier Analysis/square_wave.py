# square_wave_instructor.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from numpy.fft import fft, ifft


def plot_samples(ax, ts, ys):
    num_samples = ts.size
    ax.plot(ts, ys, color="lightgray", linewidth=1)
    ax.scatter(ts, ys, color="black", marker=".", s=10.0, zorder=2)
    ax.set_title(f"Sampled Wave ({num_samples} samples)")
    ax.set_xlabel("scaled time", loc="right")
    ax.set_ylabel("amplitude")


def plot_dft(ax, ct):
    num_terms = 40

    # fmt: off
    ax.bar(range(0, num_terms), ct.real[:num_terms],
        color="blue", label="cosine", zorder=2)
    ax.bar(range(0, num_terms), -ct.imag[:num_terms],
        color="red",  label="sine", zorder=2)    
    # fmt: on

    ax.grid(which="major", axis="x", color="black", linewidth=1)
    ax.grid(which="minor", axis="x", color="lightgray", linewidth=1)
    ax.grid(which="major", axis="y", color="black", linewidth=1)
    ax.grid(which="minor", axis="y", color="lightgray", linewidth=1)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title("Fast Fourier Transform")
    ax.set_xlabel("frequency", loc="right")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")


def plot_idft(ax, ts, yr):
    num_samples = ts.size
    ax.plot(ts, yr, color="purple")
    ax.set_title(f"Inverse FFT ({num_samples} samples)")
    ax.set_xlabel("scaled time", loc="right")
    ax.set_ylabel("amplitude")


def plot_power_spectrum(ax, ct):
    num_terms = 40
    ax.bar(
        range(0, num_terms), abs(ct[:num_terms]), color="green", label="sine", zorder=2
    )
    ax.grid(which="major", axis="x", color="black", linewidth=1)
    ax.grid(which="minor", axis="x", color="lightgray", linewidth=1)
    ax.grid(which="major", axis="y", color="black", linewidth=1)
    ax.grid(which="minor", axis="y", color="lightgray", linewidth=1)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title("Power Spectrum")
    ax.set_xlabel("frequency", loc="right")
    ax.set_ylabel(r"$\Vert amplitude \Vert$")


def main(file_name):
    file_path = Path(__file__).parent / file_name
    samples = np.genfromtxt(file_path, delimiter=",")
    ts, ys = samples.T

    ct = 2 / len(ys) * fft(ys)
    yr = len(ys) / 2 * ifft(ct)
    ct[0] /= 2  # DC value should NOT be doubled

    plt.figure(
        Path(__file__).name + f" ({file_name})",
        figsize=(12, 8),
    )

    plot_samples(plt.subplot(2, 2, 1), ts, ys)
    plot_dft(plt.subplot(2, 2, 2), ct)
    plot_idft(plt.subplot(2, 2, 3), ts, np.real(yr))
    plot_power_spectrum(plt.subplot(2, 2, 4), ct)

    plt.tight_layout()
    plt.show()


file_name = "samples_square.csv"
main(file_name)

"""
Put your comments here
"""
