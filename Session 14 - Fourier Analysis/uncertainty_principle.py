#!/usr/bin/env -S uv run
"""uncertainty_principle.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


def gaussian_wavefunction(xs: NDArray, sigma: float) -> NDArray:
    """Return a normalized Gaussian wavefunction psi(x)"""
    psi = np.exp(-(xs**2) / (4 * sigma**2))

    dx = xs[1] - xs[0]
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    return psi / norm


def position_probability_density(psi_x: NDArray) -> NDArray:
    """Return the position probability density |psi(x)|^2"""
    return np.abs(psi_x) ** 2


def momentum_probability_density(psi_x: NDArray, dx: float) -> tuple[NDArray, NDArray]:
    """Return wave numbers k and the probability density |phi(k)|^2"""

    # Number of position-space samples in psi(x)
    num_samples = psi_x.size

    # Build the corresponding wave-number values k
    # fftfreq gives cycles/unit length; multiply by 2π to get radians/unit length
    # fftshift moves negative k values to the left and positive k values to the right
    ks = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(num_samples, d=dx))

    # Spacing between neighboring k values
    dk = ks[1] - ks[0]

    # Compute phi(k), the Fourier transform of psi(x)
    # The dx factor makes the discrete sum approximate the continuous integral
    # The 1/sqrt(2π) factor matches the symmetric Fourier transform convention
    # fftshift centers the zero-frequency component in the plotted spectrum
    phi_k = (
        dx / np.sqrt(2 * np.pi) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_x)))
    )

    # Convert complex amplitudes phi(k) into probability density |phi(k)|^2
    prob_k = np.abs(phi_k) ** 2

    # Normalize so the total area under the momentum probability curve is 1
    prob_k /= np.sum(prob_k) * dk

    return ks, prob_k


def build_distributions(xs: NDArray, sigma: float) -> tuple[NDArray, NDArray, NDArray]:
    """Build normalized position and momentum probability densities"""
    dx = xs[1] - xs[0]

    psi_x = gaussian_wavefunction(xs, sigma)
    prob_x = position_probability_density(psi_x)
    ks, prob_k = momentum_probability_density(psi_x, dx)

    return prob_x, ks, prob_k


def sigma_generator():
    """Shrink from the starting width, then return to it"""
    sigma_start = 0.25
    sigma_stop = 0.015
    num_frames_each_way = 240

    shrinking = np.linspace(
        sigma_start,
        sigma_stop,
        num_frames_each_way,
        endpoint=True,
    )

    widening = np.linspace(
        sigma_stop,
        sigma_start,
        num_frames_each_way,
        endpoint=True,
    )[1:]

    yield from shrinking
    yield from widening


def plot_position(ax, xs: NDArray, prob_x: NDArray):
    """Plot the initial position probability density"""
    sigma_stop = 0.015
    prob_x_max, _, _ = build_distributions(xs, sigma_stop)

    (line,) = ax.plot(xs, prob_x, animated=True)

    ax.set_title("Particle Location")
    ax.set_xlabel("Position $x$", loc="right")
    ax.set_ylabel(r"Probability Density $|\psi(x)|^2$")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 1.15 * np.max(prob_x_max))

    return line


def plot_momentum(ax, ks: NDArray, prob_k: NDArray):
    """Plot the initial momentum probability density"""
    (line,) = ax.plot(ks, prob_k, animated=True)

    ax.set_title("Particle Frequency / Momentum")
    ax.set_xlabel(r"Wave number $k$", loc="right")
    ax.set_ylabel(r"Probability Density $|\phi(k)|^2$")
    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 1.15 * np.max(prob_k))

    return line


def draw_frame(sigma: float):
    """Update both probability distributions for the current sigma"""
    prob_x, ks, prob_k = build_distributions(xs, sigma)

    wave_position.set_data(xs, prob_x)
    wave_momentum.set_data(ks, prob_k)

    return wave_position, wave_momentum


def main():
    global xs, wave_position, wave_momentum, anim

    xs = np.linspace(-1, 1, 1000, endpoint=False)

    sigma_start = 0.25
    prob_x, ks, prob_k = build_distributions(xs, sigma_start)

    plt.figure(Path(__file__).name, figsize=(12, 4))

    ax_position = plt.subplot(1, 2, 1)
    ax_momentum = plt.subplot(1, 2, 2)

    wave_position = plot_position(ax_position, xs, prob_x)
    wave_momentum = plot_momentum(ax_momentum, ks, prob_k)

    anim = FuncAnimation(
        ax_position.figure,
        draw_frame,
        frames=sigma_generator,
        interval=25,
        cache_frame_data=False,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
