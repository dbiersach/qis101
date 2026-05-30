#!/usr/bin/env -S uv run
"""uncertainty_principle.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


def gaussian_wavefunction(xs: NDArray, sigma: float) -> NDArray:
    """Return a normalized Gaussian wavefunction psi(x)"""
    # This is the position-space wavefunction psi(x)
    # for a Gaussian distribution centered at x=0 with width sigma
    psi_x = np.exp(-(xs**2) / (4 * sigma**2))
    # Compute the spacing between x values for normalization
    dx = xs[1] - xs[0]
    # Normalize the wavefunction so that the integral of |psi(x)|^2 is 1
    psi_x /= np.sqrt(np.sum(np.abs(psi_x) ** 2) * dx)
    return psi_x


def position_pdf(psi_x: NDArray) -> NDArray:
    """Return the position probability density |psi(x)|^2"""
    return np.abs(psi_x) ** 2


def momentum_pdf(psi_x: NDArray, dx: float) -> tuple[NDArray, NDArray]:
    """Return wave numbers k and the momentum probability density |phi(k)|^2"""
    # Compute the momentum-space wavefunction phi(k) using the Fourier transform of psi(x)
    phi_k = (
        dx / np.sqrt(2 * np.pi) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_x)))
    )
    # Compute the wave numbers k corresponding to the FFT frequencies
    ks = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(psi_x.size, d=dx))
    # Compute the spacing between k values for normalization
    dk = ks[1] - ks[0]
    # Compute the momentum probability density |phi(k)|^2
    momentum_pdf_k = np.abs(phi_k) ** 2
    # Normalize the momentum probability density
    momentum_pdf_k /= np.sum(momentum_pdf_k) * dk
    return ks, momentum_pdf_k


def build_distributions(xs: NDArray, sigma: float) -> tuple[NDArray, NDArray, NDArray]:
    """Build normalized position and momentum probability densities"""
    psi_x = gaussian_wavefunction(xs, sigma)
    position_pdf_x = position_pdf(psi_x)
    # Compute the spacing between x values for normalization
    dx = xs[1] - xs[0]
    # Compute the momentum probability density and corresponding wave numbers
    ks, momentum_pdf_k = momentum_pdf(psi_x, dx)
    return position_pdf_x, ks, momentum_pdf_k


def sigma_generator():
    """Shrink from the starting width, then return to it"""
    sigma_start = 0.25
    sigma_stop = 0.015
    num_frames_each_way = 240
    shrinking = np.linspace(sigma_start, sigma_stop, num_frames_each_way)
    widening = np.linspace(sigma_stop, sigma_start, num_frames_each_way)[1:]
    yield from shrinking
    yield from widening


def plot_position(ax, xs: NDArray, position_pdf_x: NDArray):
    """Plot the initial position probability density"""
    sigma_stop = 0.015
    position_pdf_max, _, _ = build_distributions(xs, sigma_stop)
    (line,) = ax.plot(xs, position_pdf_x, animated=True)
    ax.set_title("Position Distribution")
    ax.set_xlabel("Position $x$", loc="right")
    ax.set_ylabel(r"Probability Density $|\psi(x)|^2$")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 1.15 * np.max(position_pdf_max))
    return line


def plot_momentum(ax, ks: NDArray, momentum_pdf_k: NDArray):
    """Plot the initial momentum probability density"""
    (line,) = ax.plot(ks, momentum_pdf_k, animated=True)
    ax.set_title("Momentum Distribution")
    ax.set_xlabel(r"Wave number $k$", loc="right")
    ax.set_ylabel(r"Probability Density $|\phi(k)|^2$")
    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 1.15 * np.max(momentum_pdf_k))
    return line


def draw_frame(sigma: float):
    """Update both probability distributions for the current sigma"""
    position_pdf_x, ks, momentum_pdf_k = build_distributions(xs, sigma)
    wave_position.set_data(xs, position_pdf_x)
    wave_momentum.set_data(ks, momentum_pdf_k)
    return wave_position, wave_momentum


def main():
    global xs, wave_position, wave_momentum, anim

    xs = np.linspace(-1, 1, 1000, endpoint=False)
    sigma_start = 0.25
    position_pdf_x, ks, momentum_pdf_k = build_distributions(xs, sigma_start)

    plt.figure(Path(__file__).name, figsize=(12, 4))
    ax_position = plt.subplot(1, 2, 1)
    ax_momentum = plt.subplot(1, 2, 2)
    wave_position = plot_position(ax_position, xs, position_pdf_x)
    wave_momentum = plot_momentum(ax_momentum, ks, momentum_pdf_k)

    # fmt: off
    anim = FuncAnimation(ax_position.figure, draw_frame,
        frames=sigma_generator, interval=25,
        cache_frame_data=False, blit=True, repeat=False)
    # fmt: on

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
