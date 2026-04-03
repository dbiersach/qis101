#!/usr/bin/env -S uv run
"""fpu_spring_mass.py

Five-mass spring chain between two fixed walls, integrated with the
Yoshida 4th-order symplectic method.  Masses move longitudinally
(along the chain axis) and obey Hooke's law with nearest-neighbor
coupling — the same topology as the original Fermi-Pasta-Ulam problem,
but linear (no nonlinear correction terms).

Uses Yoshida's 4th-order symplectic integrator (1990), which composes
three Leapfrog substeps with weighted time increments:
    cbrt2 = 2^(1/3)
    c1 = c4 =  1 / (2 * (2 - cbrt2))           ≈  0.6756
    c2 = c3 = (1 - cbrt2) / (2 * (2 - cbrt2))   ≈ -0.1756
    d1 = d3 =  1 / (2 - cbrt2)                  ≈  1.3512
    d2      = -cbrt2 / (2 - cbrt2)               ≈ -1.7024
Note: c2, c3, and d2 are negative — the integrator temporarily
steps backward in the middle substep to cancel lower-order errors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks
from tqdm import tqdm

# ── Physical parameters ──────────────────────────────────────────────
N = 5  # number of masses
K = 50.0  # spring constant (N/m)
M = 1.0  # mass of each particle (kg)
ALPHA = 0.0  # FPU nonlinearity strength (0 = pure Hooke's law)
A = 1.0  # equilibrium spacing (m); walls at x=0 and x=(N+1)*A

# Preallocate padded position array shared by acceleration() and main()
_u_ext = np.empty(N + 2)
_u_ext[0] = 0.0  # left wall (fixed)
_u_ext[-1] = 0.0  # right wall (fixed)


def acceleration(u: np.ndarray) -> np.ndarray:
    """
    Compute the acceleration of each mass in the spring chain.

    Walls are fixed at displacement zero (u₀ = u_{N+1} = 0). Each mass
    experiences forces from its two nearest-neighbor springs. When ALPHA
    is zero this reduces to pure Hooke's law; nonzero ALPHA adds an FPU
    quadratic correction term.

    Parameters
    ----------
    u : ndarray, shape (N,)
        Displacement of each mass from its equilibrium position (m)

    Returns
    -------
    ndarray, shape (N,)
        Acceleration of each mass (m/s²)
    """
    _u_ext[1:-1] = u
    du_right = _u_ext[2:] - _u_ext[1:-1]  # stretch of right spring
    du_left = _u_ext[1:-1] - _u_ext[:-2]  # stretch of left spring
    f_right = K * du_right + ALPHA * K * du_right**2
    f_left = K * du_left + ALPHA * K * du_left**2
    return (f_right - f_left) / M


def integrate(
    u0: np.ndarray, v0: np.ndarray, tf: float, ts: int, desc: str = "Integrating"
) -> tuple:
    """
    Run the Yoshida 4th-order symplectic integration.

    Parameters
    ----------
    u0 : ndarray, shape (N,)
        Initial displacements from equilibrium (m)
    v0 : ndarray, shape (N,)
        Initial velocities (m/s)
    tf : float
        Final time (s)
    ts : int
        Number of time steps

    Returns
    -------
    t_hist : ndarray, shape (ts,)
        Time array (s)
    u_hist : ndarray, shape (ts, N)
        Displacement of each mass at each time step (m)
    """
    dt = tf / ts

    # ── Yoshida 4th-order symplectic coefficients (1990) ─────────────
    cbrt2 = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (2.0 - cbrt2))
    c2 = (1.0 - cbrt2) / (2.0 * (2.0 - cbrt2))
    cs = [c1, c2, c2, c1]
    d1 = 1.0 / (2.0 - cbrt2)
    d2 = -cbrt2 / (2.0 - cbrt2)
    ds = [d1, d2, d1]

    t_hist = np.zeros(ts)
    u_hist = np.zeros((ts, N))
    u = u0.copy()
    v = v0.copy()
    u_hist[0] = u

    for step in tqdm(range(1, ts), desc=desc):
        u = u + cs[0] * v * dt
        for j in range(3):
            v = v + ds[j] * acceleration(u) * dt
            u = u + cs[j + 1] * v * dt
        t_hist[step] = step * dt
        u_hist[step] = u.copy()

    return t_hist, u_hist


def main() -> None:
    # ── Initial conditions ───────────────────────────────────────────
    u0 = np.zeros(N)  # displacements from equilibrium (m)
    v0 = np.zeros(N)  # velocities (m/s)
    u0[0] = 0.3  # pluck first mass forward from equilibrium

    # ── Short run for time-domain plots (0–10s) ──────────────────────
    tf_plot, ts_plot = 10.0, 100_000
    t_hist, u_hist = integrate(u0, v0, tf_plot, ts_plot, desc="Time Domain Integration")

    # ── Long run for FFT (0–100s gives 0.01 Hz frequency resolution) ─
    tf_fft, ts_fft = 100.0, 200_000
    _, u_hist_fft = integrate(
        u0, v0, tf_fft, ts_fft, desc="Frequency Domain Integration"
    )
    dt_fft = tf_fft / ts_fft

    # ── Plotting ─────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), num=Path(__file__).name)
    ax1.sharex(ax2)
    colors = ["crimson", "royalblue", "darkorange", "forestgreen", "mediumpurple"]
    ax1.tick_params(axis="x", labelbottom=True)
    ax1.set_xlabel("Time (sec)")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1.0))

    # Upper panel: displacement from equilibrium vs time (short run)
    for i in range(N):
        ax1.plot(t_hist, u_hist[:, i], lw=1.5, color=colors[i], label=f"mass {i + 1}")
    ax1.set_ylabel("Displacement from equilibrium (m)")
    ax1.set_title(
        rf"FPU Spring-Mass Chain — 4th-Order Symplectic (Yoshida)  $\alpha={ALPHA}$"
    )
    ax1.legend(loc="upper right", fontsize=8, framealpha=1.0, facecolor="white")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.grid(True)

    # Middle panel: absolute position vs time (short run) with mass 1 zero-crossings
    for i in range(N):
        eq_pos = (i + 1) * A
        ax2.plot(
            t_hist,
            eq_pos + u_hist[:, i],
            lw=1.5,
            color=colors[i],
            label=f"mass {i + 1} (eq = {eq_pos:.0f} m)",
        )
    ax2.axhline(0, color="black", lw=2, ls="--", label="left wall")
    ax2.axhline((N + 1) * A, color="black", lw=2, ls="--", label="right wall")
    u1 = u_hist[:, 0]
    zero_crossings = np.where(np.diff(np.sign(u1)))[0]
    # for idx in zero_crossings:
    #    ax2.axvline(t_hist[idx], color="black", ls=":")
    ax2.vlines(t_hist[zero_crossings], ymin=0.5, ymax=5.5, color="black", ls=":")
    ax2.set_xlabel("Time (sec)")
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax2.set_ylabel("Absolute x position (m)")
    ax2.legend(loc="upper right", fontsize=8, framealpha=1.0, facecolor="white")
    ax2.grid(True)

    # Bottom panel: grouped bar chart of peak power at each normal mode frequency
    freqs = fftfreq(ts_fft, dt_fft)
    pos_mask = freqs > 0
    window = np.hanning(ts_fft)
    window_gain = window.sum()

    ref_freqs = freqs[pos_mask]

    # Compute spectra for all masses and find the five normal mode peaks
    all_spectra_db = []
    for i in range(N):
        signal = u_hist_fft[:, i]
        spectrum = np.abs(fft(signal * window)) * 2 / window_gain
        power_db = 20 * np.log10(np.maximum(spectrum, 1e-15))
        all_spectra_db.append(power_db[pos_mask])  # keep positive freqs only

    # Find peaks from measured FFT data — works correctly for any alpha,
    # since actual mode frequencies shift with nonlinearity
    all_spectra_array = np.array(all_spectra_db)
    ref_spectrum = all_spectra_array.max(axis=0)  # envelope across all masses
    freq_resolution = ref_freqs[1] - ref_freqs[0]
    min_separation = int(0.2 / freq_resolution)
    peak_indices, _ = find_peaks(ref_spectrum, height=-40, distance=min_separation)
    peak_indices = peak_indices[np.argsort(ref_spectrum[peak_indices])[::-1]][:N]
    peak_indices = np.sort(peak_indices)
    peak_freqs = ref_freqs[peak_indices]

    # Build grouped bar chart: N masses × N mode frequencies
    bar_width = 0.012
    offsets = np.linspace(-(N - 1) / 2, (N - 1) / 2, N) * bar_width * 1.1
    bottom_db = -50
    for i in range(N):
        peak_powers = np.array([all_spectra_db[i][idx] for idx in peak_indices])
        bar_heights = peak_powers - bottom_db  # height above the bottom baseline
        ax3.bar(
            peak_freqs + offsets[i],
            bar_heights,
            width=bar_width,
            color=colors[i],
            label=f"mass {i + 1}",
            bottom=bottom_db,
            align="center",
        )
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power (dB)")
    ax3.set_ylim(-50, -10)
    ax3.set_xticks(peak_freqs)
    ax3.set_xticklabels([f"{f:.3f} Hz" for f in peak_freqs], fontsize=8)
    ax3.set_title("Normal Mode Power by Mass", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8, framealpha=1.0, facecolor="white")
    ax3.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
