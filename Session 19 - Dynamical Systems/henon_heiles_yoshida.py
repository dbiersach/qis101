#!/usr/bin/env -S uv run
"""henon_heiles_yoshida.py

Simulate the Hénon-Heiles Hamiltonian system with Yoshida's 4th-order
symplectic integrator and plot:

1. q1(t) and p1(t)
2. Configuration-space trajectories in the (q1, q2) plane
3. A Poincaré section in the (q2, p2) plane at q1 = 0, ṗ1 > 0
4. Energy drift |ΔH(t)| confirming symplectic conservation

Reference: Hénon & Heiles (1964), The Astronomical Journal, 69, 73.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

LAMBDA = 1.0  # cubic coupling constant (canonical Hénon-Heiles value)


def hamiltonian(
    q1: float | NDArray[np.float64],
    q2: float | NDArray[np.float64],
    p1: float | NDArray[np.float64],
    p2: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """
    Evaluate the Hénon-Heiles Hamiltonian H = T + V.

    H = (p1² + p2²)/2 + (q1² + q2²)/2 + λ(q1²q2 − q2³/3)

    Parameters
    ----------
    q1 : float or NDArray[np.float64]
        Generalized coordinate 1
    q2 : float or NDArray[np.float64]
        Generalized coordinate 2
    p1 : float or NDArray[np.float64]
        Conjugate momentum to q1
    p2 : float or NDArray[np.float64]
        Conjugate momentum to q2

    Returns
    -------
    float or NDArray[np.float64]
        Total energy of the system
    """
    kinetic = 0.5 * (p1**2 + p2**2)
    potential = 0.5 * (q1**2 + q2**2) + LAMBDA * (q1**2 * q2 - q2**3 / 3.0)
    return kinetic + potential


def force(q1: float, q2: float) -> tuple[float, float]:
    """
    Return the generalized forces (dp1/dt, dp2/dt) at position (q1, q2).

    Derived from Hamilton's equations dpi/dt = −∂H/∂qi:
        dp1/dt = −q1 − 2λq1q2
        dp2/dt = −q2 − λ(q1² − q2²)

    Parameters
    ----------
    q1 : float
        Generalized coordinate 1
    q2 : float
        Generalized coordinate 2

    Returns
    -------
    f1 : float
        Force on coordinate 1 (= dp1/dt)
    f2 : float
        Force on coordinate 2 (= dp2/dt)
    """
    f1 = -q1 - 2.0 * LAMBDA * q1 * q2
    f2 = -q2 - LAMBDA * (q1**2 - q2**2)
    return f1, f2


def solve_yoshida4(
    q1_0: float,
    q2_0: float,
    p1_0: float,
    p2_0: float,
    t_final: float,
    dt: float,
    desc: str = "Integrating",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Integrate one trajectory with Yoshida's 4th-order symplectic method.

    Constructed from three Verlet substeps whose coefficients cancel errors of
    order dt² and dt³, giving a global error of order dt⁴ (Forest & Ruth 1990
    / Yoshida 1990).  Being symplectic, the method preserves the phase-space
    volume element dq1∧dq2∧dp1∧dp2 at every step, bounding |ΔH| for all time.

    Parameters
    ----------
    q1_0 : float
        Initial generalized coordinate 1
    q2_0 : float
        Initial generalized coordinate 2
    p1_0 : float
        Initial conjugate momentum to q1
    p2_0 : float
        Initial conjugate momentum to q2
    t_final : float
        Total integration time (s)
    dt : float
        Fixed time step size (s)
    desc : str
        Label shown on the tqdm progress bar for this trajectory

    Returns
    -------
    t : NDArray[np.float64], shape (n_steps,)
        Time array (s)
    q1, q2 : NDArray[np.float64], shape (n_steps,)
        Generalized coordinates at each time step
    p1, p2 : NDArray[np.float64], shape (n_steps,)
        Conjugate momenta at each time step
    """
    # Yoshida coefficients - identical for any separable H = T(p) + V(q)
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)
    c = np.array([w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0])
    d = np.array([w1, w0, w1])

    n_steps = int(t_final / dt)
    t = np.arange(n_steps, dtype=np.float64) * dt
    q1 = np.zeros(n_steps)
    q2 = np.zeros(n_steps)
    p1 = np.zeros(n_steps)
    p2 = np.zeros(n_steps)
    q1[0], q2[0], p1[0], p2[0] = q1_0, q2_0, p1_0, p2_0

    for i in tqdm(range(n_steps - 1), desc=desc):
        r1, r2, s1, s2 = q1[i], q2[i], p1[i], p2[i]
        for j in range(3):
            r1 += c[j] * s1 * dt  # position drift
            r2 += c[j] * s2 * dt
            f1, f2 = force(r1, r2)
            s1 += d[j] * f1 * dt  # momentum kick at updated position
            s2 += d[j] * f2 * dt
        r1 += c[3] * s1 * dt  # final position half-drift
        r2 += c[3] * s2 * dt
        q1[i + 1], q2[i + 1], p1[i + 1], p2[i + 1] = r1, r2, s1, s2

    return t, q1, q2, p1, p2


def poincare_section(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return (q2, p2) where the orbit crosses q1 = 0 with p1 > 0.

    Linear interpolation between bracketing steps gives sub-step crossing
    accuracy without requiring a finer integration grid.

    Parameters
    ----------
    q1, q2, p1, p2 : NDArray[np.float64]
        Phase-space trajectory arrays (equal length)

    Returns
    -------
    sec_q2 : NDArray[np.float64]
        q2 values at section crossings
    sec_p2 : NDArray[np.float64]
        p2 values at section crossings
    """
    sec_q2, sec_p2 = [], []
    for i in range(len(q1) - 1):
        if q1[i] < 0.0 and q1[i + 1] >= 0.0 and p1[i] > 0.0:
            alpha = -q1[i] / (q1[i + 1] - q1[i])  # fractional step to q1 = 0
            sec_q2.append(q2[i] + alpha * (q2[i + 1] - q2[i]))
            sec_p2.append(p2[i] + alpha * (p2[i + 1] - p2[i]))
    return np.array(sec_q2), np.array(sec_p2)


def initial_conditions(
    energy: float, n_traj: int = 6
) -> list[tuple[float, float, float, float]]:
    """
    Generate n_traj initial conditions on the energy surface H = energy.

    Sets q2 = 0, p2 = 0 and distributes p1 across [0.05·p1_max, 0.92·p1_max],
    computing q1 from energy conservation: q1 = sqrt(2E − p1²).  Endpoints are
    excluded to avoid degenerate fixed-point orbits.

    Parameters
    ----------
    energy : float
        Target Hamiltonian value (must satisfy 0 < energy < 1/6)
    n_traj : int
        Number of distinct trajectories to generate

    Returns
    -------
    list of (q1_0, q2_0, p1_0, p2_0) tuples
    """
    p1_max = np.sqrt(2.0 * energy)
    p1_vals = np.linspace(0.05 * p1_max, 0.92 * p1_max, n_traj)
    ics = []
    for p1_0 in p1_vals:
        q1_0 = np.sqrt(max(2.0 * energy - p1_0**2, 0.0))
        ics.append((q1_0, 0.0, p1_0, 0.0))
    return ics


def main() -> None:
    """Run the simulation and show the four diagnostic plots."""
    energy = 1 / 12  # subcritical: try 1/8 for mixed regime, 1/6 for chaos
    t_final = 5000.0
    dt = 0.01  # 500 000 steps; |ΔH| stays below ~1e-9 throughout
    n_traj = 6

    stride_ts = 5  # time-series:  one point per 0.05 s
    stride_config = 50  # config-space: 10 000 pts per trajectory
    stride_energy = 250  # energy drift:  2 000 pts over full run

    ics = initial_conditions(energy, n_traj)
    cmap = plt.get_cmap("tab10")

    trajs = []
    for k, (q1_0, q2_0, p1_0, p2_0) in enumerate(ics):
        desc = f"Traj {k + 1}/{n_traj} (q1\u2080={q1_0:.3f}, p1\u2080={p1_0:.3f})"
        trajs.append(solve_yoshida4(q1_0, q2_0, p1_0, p2_0, t_final, dt, desc))

    # Figure 1: time series - first 200 s, fine stride for smooth curves
    t, q1, q2, p1, p2 = trajs[3]
    show = (t <= 200.0) & (np.arange(len(t)) % stride_ts == 0)

    plt.figure(f"{Path(__file__).name} - Time Series")
    plt.plot(t[show], q1[show], color="crimson", lw=1.2, label=r"$q_1(t)$")
    plt.plot(t[show], p1[show], color="forestgreen", lw=1.2, label=r"$p_1(t)$")
    plt.title(r"Hénon-Heiles: Time Series  ($E = 1/12$, Yoshida 4th-Order)")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$q_1$,  $p_1$")
    plt.legend(framealpha=1.0, facecolor="white")
    plt.grid(True)
    plt.tight_layout()

    # Figure 2: configuration space - scatter avoids false diagonals from striding
    plt.figure(f"{Path(__file__).name} - Configuration Space")
    for k, (t, q1, q2, p1, p2) in enumerate(trajs):
        q1_0, _, p1_0, _ = ics[k]
        label = f"Traj {k + 1}: $q_1^0={q1_0:.3f}$, $p_1^0={p1_0:.3f}$"
        plt.scatter(
            q1[::stride_config],
            q2[::stride_config],
            s=0.25,
            marker=".",
            color=cmap(k),
            label=label,
        )
    plt.title(r"Hénon-Heiles: Configuration Space  ($E = 1/12$)")
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.legend(
        loc="upper right",
        fontsize=7,
        framealpha=1.0,
        facecolor="white",
        markerscale=3,
        title="Initial conditions",
        title_fontsize=7,
    )
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    # Figure 3: Poincaré section - full resolution, every crossing recorded
    plt.figure(f"{Path(__file__).name} - Poincaré Section")
    for k, (t, q1, q2, p1, p2) in enumerate(trajs):
        sq2, sp2 = poincare_section(q1, q2, p1, p2)
        if len(sq2) == 0:
            continue
        q1_0, _, p1_0, _ = ics[k]
        label = f"Traj {k + 1}: $q_1^0={q1_0:.3f}$, $p_1^0={p1_0:.3f}$"
        plt.scatter(sq2, sp2, s=4, color=cmap(k), label=label)
    plt.title(r"Hénon-Heiles: Poincaré Section  $q_1=0,\ \dot{q}_1>0\ (E=1/12)$")
    plt.xlabel(r"$q_2$")
    plt.ylabel(r"$p_2$")
    plt.legend(
        loc="upper right",
        fontsize=7,
        framealpha=1.0,
        facecolor="white",
        markerscale=2.5,
        title="Initial conditions",
        title_fontsize=7,
    )
    plt.grid(True)
    plt.tight_layout()

    # Figure 4: energy drift - 2 000 pts shows the oscillating envelope clearly
    t, q1, q2, p1, p2 = trajs[3]
    e0 = hamiltonian(q1[0], q2[0], p1[0], p2[0])
    delta_e = np.abs(hamiltonian(q1, q2, p1, p2) - e0)

    plt.figure(f"{Path(__file__).name} - Energy Conservation")
    plt.semilogy(t[::stride_energy], delta_e[::stride_energy], color="purple", lw=0.8)
    plt.title(r"Hénon-Heiles: Energy Drift  $|\Delta H(t)|$  (Yoshida 4th-Order)")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$|H(t) - H_0|$")
    plt.grid(True, which="both")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
