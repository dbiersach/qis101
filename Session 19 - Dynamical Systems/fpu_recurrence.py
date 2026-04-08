#!/usr/bin/env -S uv run
"""fpu_recurrence.py

Fermi-Pasta-Ulam alpha model: 32 masses connected by nonlinear springs
between two fixed walls.  The force law includes a quadratic correction:

    F = k * (du) + alpha * k * (du)^2

where du = u_{i+1} - u_i is the stretch of each spring.

Integrated with the Yoshida 4th-order symplectic method.

The upper panel tracks energy in each normal mode of the linear chain
over time, showing the famous FPU recurrence - energy flows from the
initially excited mode into other modes, then flows back instead of
thermalizing as Fermi expected.

The lower panel is a sanity check: total energy summed across all modes
should remain constant, confirming the symplectic integrator is not
introducing artificial energy drift.

Normal modes for N masses between fixed walls:
    Q_j = sqrt(2/(N+1)) * sum_i [ u_i * sin(j * pi * i / (N+1)) ]
    P_j = sqrt(2/(N+1)) * sum_i [ v_i * sin(j * pi * i / (N+1)) ]
    omega_j = 2 * sqrt(k/m) * sin(j * pi / (2*(N+1)))
    E_j = (P_j^2 + omega_j^2 * Q_j^2) / 2
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ── Physical parameters ──────────────────────────────────────────────
N = 32  # number of masses (more masses = cleaner recurrence)
K = 1.0  # spring constant (N/m)
M = 1.0  # mass of each particle (kg)
ALPHA = 0.7  # FPU nonlinearity strength
A = 1.0  # equilibrium spacing (m)

# Preallocate padded position array shared by acceleration() and main()
_u_ext = np.empty(N + 2)
_u_ext[0] = 0.0  # left wall (fixed)
_u_ext[-1] = 0.0  # right wall (fixed)

# ── Precompute normal mode basis and frequencies ─────────────────────
# mode_matrix[j, i] = sqrt(2/(N+1)) * sin((j+1) * pi * (i+1) / (N+1))
_j_idx = np.arange(1, N + 1).reshape(N, 1)
_i_idx = np.arange(1, N + 1).reshape(1, N)
_mode_matrix = np.sqrt(2.0 / (N + 1)) * np.sin(_j_idx * np.pi * _i_idx / (N + 1))
_omega = 2.0 * np.sqrt(K / M) * np.sin(_j_idx.ravel() * np.pi / (2 * (N + 1)))
_omega_sq = _omega**2


def acceleration(u: np.ndarray) -> np.ndarray:
    """
    Compute the FPU-alpha acceleration for each mass in the spring chain.

    Walls are fixed at displacement zero (u₀ = u_{N+1} = 0). Each mass
    experiences forces from its two nearest-neighbor springs, with a
    quadratic correction term scaled by ALPHA.

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


def mode_energies(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project positions and velocities onto normal modes and compute modal energies.

    Uses the precomputed mode matrix and angular frequencies to evaluate
    E_j = (P_j² + ω_j² Q_j²) / 2 for each normal mode j.

    Parameters
    ----------
    u : ndarray, shape (N,)
        Displacement of each mass from its equilibrium position (m)
    v : ndarray, shape (N,)
        Velocity of each mass (m/s)

    Returns
    -------
    ndarray, shape (N,)
        Energy in each normal mode (J)
    """
    Q = _mode_matrix @ u  # normal mode coordinates
    P = _mode_matrix @ v  # normal mode momenta
    return 0.5 * (P**2 + _omega_sq * Q**2)


def main() -> None:
    # ── Simulation parameters ────────────────────────────────────────
    tf = 2 * np.pi * 1000  # final time (s) - long enough to see recurrence
    ts = 1_000_000  # number of time steps
    dt = tf / ts

    # ── Initial conditions ───────────────────────────────────────────
    # Excite only normal mode 1 (lowest mode) - the classic FPU setup
    mode = 1
    amplitude = 1.0
    indices = np.arange(1, N + 1)
    u = amplitude * np.sin(mode * np.pi * indices / (N + 1))
    v = np.zeros(N)

    # ── Yoshida 4th-order symplectic coefficients (1990) ─────────────
    cbrt2 = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (2.0 - cbrt2))
    c2 = (1.0 - cbrt2) / (2.0 * (2.0 - cbrt2))
    cs = [c1, c2, c2, c1]
    d1 = 1.0 / (2.0 - cbrt2)
    d2 = -cbrt2 / (2.0 - cbrt2)
    ds = [d1, d2, d1]

    # ── Storage (sample every sample_every steps to save memory) ─────
    # Derive n_samples from sample_every so the two are always exactly
    # aligned; ts % sample_every == 0 is guaranteed by construction
    sample_every = 5_000
    n_samples = ts // sample_every
    t_hist = np.zeros(n_samples)
    energy_hist = np.zeros((n_samples, N))
    sample_idx = 0

    # ── Time integration (Yoshida 4th-order) ─────────────────────────
    for step in tqdm(range(ts), desc="Integrating"):
        if step % sample_every == 0:
            t_hist[sample_idx] = step * dt
            energy_hist[sample_idx] = mode_energies(u, v)
            sample_idx += 1

        u = u + cs[0] * v * dt
        for j in range(3):
            v = v + ds[j] * acceleration(u) * dt
            u = u + cs[j + 1] * v * dt

    # ── Plotting ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), num=Path(__file__).name)

    # Upper panel: energy in first several modes vs time
    n_modes_plot = 6
    colors = [
        "crimson",
        "royalblue",
        "darkorange",
        "forestgreen",
        "mediumpurple",
        "saddlebrown",
    ]
    for j in range(n_modes_plot):
        ax1.plot(
            t_hist, energy_hist[:, j], lw=1.5, color=colors[j], label=f"mode {j + 1}"
        )
    ax1.set_ylabel("Energy in mode (J)")
    ax1.set_title(
        rf"FPU Recurrence ({N} masses, $\alpha$ = {ALPHA}),"
        " Yoshida 4th-order symplectic"
    )
    ax1.legend(loc="center right", fontsize=8, framealpha=1.0, facecolor="white")
    ax1.grid(True)

    # Lower panel: total energy - should be flat (symplectic conservation check)
    total_energy = np.sum(energy_hist, axis=1)
    ax2.plot(t_hist, total_energy, lw=1.0, color="black")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Total energy (J)")
    ax2.set_title("Total energy (symplectic conservation check)")
    ax2.grid(True)

    # Keep y-axis tight around the actual value to reveal any drift
    e_mean = np.mean(total_energy)
    e_range = max(np.ptp(total_energy) * 5, e_mean * 1e-6)
    ax2.set_ylim(e_mean - e_range, e_mean + e_range)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
