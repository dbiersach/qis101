# fourier_discrete.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


def dft(ts, fs):
    num_samples = ts.size  # ts = sample time
    num_terms = int(num_samples / 2)  # Nyquist limit
    # ct = complex terms of the DFT
    ct = np.zeros(num_terms, dtype=complex)
    for k in range(0, num_terms):  # k = filter wave number
        for n in range(0, num_samples):  # n = sample number
            ct[k] += fs[n] * np.exp(complex(0, (k * ts[n])))
    ct = ct * 2 / num_samples
    ct[0] /= 2  # DC value should NOT be doubled
    return ct


def idft(ts, ct):
    num_samples = ts.size  # ts = sample time
    num_terms = ct.size
    # fr = reconstructed fs(t) values
    fr = np.zeros(num_samples, dtype=complex)
    for n in range(0, num_samples):  # n = sample number
        for k in range(0, num_terms):  # k = filter wave number
            fr[n] += ct[k] * np.exp(complex(0, -(k * ts[n])))
    return fr


def plot_samples(ax, ts, fs):
    num_samples = ts.size
    ax.plot(ts, fs, color="lightgray", linewidth=1)
    ax.scatter(ts, fs, color="black", marker=".", s=10.0, zorder=2)
    ax.set_title(f"Sampled Wave ({num_samples} samples)")
    ax.set_xlabel("scaled time", loc="right")
    ax.set_ylabel("amplitude")


def plot_dft(ax, ct):
    num_terms = 40

    # fmt: off
    ax.bar(range(0, num_terms), ct.real[:num_terms],
        color="blue", label="cosine", zorder=2)
    ax.bar(range(0, num_terms), ct.imag[:num_terms],
        color="red",  label="sine", zorder=2)
    # fmt: on

    ax.grid(which="major", axis="x", color="black", linewidth=1)
    ax.grid(which="minor", axis="x", color="lightgray", linewidth=1)
    ax.grid(which="major", axis="y", color="black", linewidth=1)
    ax.grid(which="minor", axis="y", color="lightgray", linewidth=1)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title("Discrete Fourier Transform")
    ax.set_xlabel("frequency", loc="right")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")


def plot_idft(ax, ts, fr):
    num_samples = ts.size
    ax.plot(ts, fr, color="purple")
    ax.set_title(f"Inverse DFT ({num_samples} samples)")
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
    # ts = sample time, fs = sample value
    ts, fs = np.genfromtxt(file_path, delimiter=",", unpack=True)

    ct = dft(ts, fs)  # ct = complex terms of the DFT
    fr = idft(ts, ct)  # fr = reconstructed fs(t) values

    plt.figure(
        Path(__file__).name + f" ({file_name})",
        figsize=(12, 8),
    )

    # subplot's 3 params: # of rows, # of cols, which subplot #
    plot_samples(plt.subplot(2, 2, 1), ts, fs)
    plot_dft(plt.subplot(2, 2, 2), ct)
    plot_idft(plt.subplot(2, 2, 3), ts, np.real(fr))
    plot_power_spectrum(plt.subplot(2, 2, 4), ct)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main("samples.csv")
