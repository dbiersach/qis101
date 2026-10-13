#!/usr/bin/env -S uv run
"""linac_electron_bremsstrahlung_direction.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.constants import c, e, m_e
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize_scalar

# An electron leaves a linear accelerator (LINAC) with kinetic energy T and
# strikes a target. The electric field of a nucleus in the target changes the
# electron's velocity abruptly, and that sudden change radiates. This is
# bremsstrahlung ("braking radiation").
#
# Classically, a charge whose velocity jumps from beta_in to beta_out radiates
# an energy per unit frequency per unit solid angle of
#
#   d^2 I / (dw dOmega) = q^2 / (16 pi^3 eps0 c) * |A(beta_out) - A(beta_in)|^2
#
#   where   A(beta) = n x (n x beta) / (1 - n . beta)
#
# and n is the unit vector from the collision toward the observer. When the
# electron stops dead (beta_out = 0) this reduces to
#
#   d^2 I / (dw dOmega) = q^2 / (16 pi^3 eps0 c) * beta^2 sin^2(theta) / (1 - beta cos(theta))^2
#
# with theta measured from the incoming beam. The constant in front only sets
# the overall scale, so every pattern is plotted normalized to its own peak.

# Electron rest energy in MeV
REST_ENERGY_MEV = m_e * c**2 / (1e6 * e)


def speed_from_kinetic_energy(kinetic_energy_mev: float) -> tuple[float, float]:
    """
    Convert an electron kinetic energy to its speed and Lorentz factor.

    Parameters
    ----------
    kinetic_energy_mev : float
        Kinetic energy in MeV.

    Returns
    -------
    tuple[float, float]
        (beta, gamma) where beta = v / c and gamma = 1 / sqrt(1 - beta^2).
    """
    gamma = 1 + kinetic_energy_mev / REST_ENERGY_MEV
    beta = np.sqrt(1 - 1 / gamma**2)
    return float(beta), float(gamma)


def velocity_term(n: np.ndarray, beta_vec: np.ndarray) -> np.ndarray:
    """
    Evaluate A(beta) = n x (n x beta) / (1 - n . beta) for one velocity.

    Parameters
    ----------
    n : np.ndarray
        Unit observation vectors, shape (..., 3).
    beta_vec : np.ndarray
        Velocity divided by c as a 3-vector.

    Returns
    -------
    np.ndarray
        The vector A(beta) at every observation direction, shape (..., 3).
    """
    n_dot_beta = n @ beta_vec
    # n x (n x beta) = n (n . beta) - beta
    transverse = n * n_dot_beta[..., np.newaxis] - beta_vec
    return transverse / (1 - n_dot_beta)[..., np.newaxis]


def radiation_pattern(
    n: np.ndarray, beta_in: np.ndarray, beta_out: np.ndarray
) -> np.ndarray:
    """
    Compute |A(beta_out) - A(beta_in)|^2 for a sudden velocity change.

    Parameters
    ----------
    n : np.ndarray
        Unit observation vectors, shape (..., 3).
    beta_in : np.ndarray
        Velocity / c before the collision, as a 3-vector.
    beta_out : np.ndarray
        Velocity / c after the collision, as a 3-vector.

    Returns
    -------
    np.ndarray
        Radiated energy per unit frequency per unit solid angle, up to the
        overall constant q^2 / (16 pi^3 eps0 c).
    """
    difference = velocity_term(n, beta_out) - velocity_term(n, beta_in)
    return np.sum(difference**2, axis=-1)


def in_plane_pattern(
    theta: np.ndarray, beta_in: np.ndarray, beta_out: np.ndarray
) -> np.ndarray:
    """
    Evaluate the pattern in the plane that holds the beam and the exit path.

    Parameters
    ----------
    theta : np.ndarray
        Angle from the incoming beam (+x axis), measured toward +y, in radians.
    beta_in, beta_out : np.ndarray
        Velocity / c before and after the collision, as 3-vectors.

    Returns
    -------
    np.ndarray
        The unnormalized pattern at each angle.
    """
    n = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
    return radiation_pattern(n, beta_in, beta_out)


def peak_angle(beta_in: np.ndarray, beta_out: np.ndarray) -> float:
    """
    Find the angle (radians, 0 to pi) where the in-plane pattern is brightest.

    A coarse scan locates the brightest lobe, then a bounded minimizer
    refines the angle, which matters at high energy where the lobe is only
    a small fraction of a degree wide.
    """
    coarse = np.linspace(0, np.pi, 7_201)  # 0.025 degree steps
    i = int(np.argmax(in_plane_pattern(coarse, beta_in, beta_out)))
    lo = coarse[max(i - 1, 0)]
    hi = coarse[min(i + 1, coarse.size - 1)]
    result = minimize_scalar(
        lambda t: -in_plane_pattern(np.array(t), beta_in, beta_out),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def energy_cone(
    beta_in: np.ndarray, beta_out: np.ndarray, fraction: float = 0.5
) -> tuple[float, float]:
    """
    Integrate the pattern over the whole sphere to measure the cone.

    Parameters
    ----------
    beta_in, beta_out : np.ndarray
        Velocity / c before and after the collision, as 3-vectors.
    fraction : float
        Share of the radiated energy the cone must enclose.

    Returns
    -------
    tuple[float, float]
        (forward_fraction, half_angle) where forward_fraction is the share of
        the energy emitted into the forward hemisphere (theta < 90 degrees)
        and half_angle (radians) is the angle from the beam that encloses the
        requested fraction of the energy.
    """
    theta = np.linspace(0, np.pi, 6_001)
    phi = np.linspace(0, 2 * np.pi, 181)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    n = np.stack(
        [np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)], axis=-1
    )
    pattern = radiation_pattern(n, beta_in, beta_out)

    # Solid angle element is sin(theta) dtheta dphi
    per_theta = trapezoid(pattern, phi, axis=1) * np.sin(theta)
    cumulative = cumulative_trapezoid(per_theta, theta, initial=0)
    total = cumulative[-1]

    forward_fraction = float(np.interp(np.pi / 2, theta, cumulative) / total)
    half_angle = float(np.interp(fraction * total, cumulative, theta))
    return forward_fraction, half_angle


def energy_label(kinetic_energy_mev: float) -> str:
    """Format a kinetic energy in keV, MeV, or GeV for titles and tables."""
    if kinetic_energy_mev < 1:
        return f"{kinetic_energy_mev * 1e3:g} keV"
    if kinetic_energy_mev < 1e3:
        return f"{kinetic_energy_mev:g} MeV"
    return f"{kinetic_energy_mev / 1e3:g} GeV"


def main():
    # TODO: Change the electron kinetic energies to compare (MeV)
    kinetic_energies_mev = [0.01, 0.1, 1.0, 10.0, 100.0]

    # TODO: Change what the collision does to the electron. An exit speed
    # fraction of 0 stops it dead. A fraction of 1 keeps its speed and only
    # turns it through the exit angle, measured from the beam toward +y.
    exit_speed_fraction = 0.0
    exit_angle_deg = 0.0

    exit_angle = np.radians(exit_angle_deg)

    print(f"Electron rest energy: {REST_ENERGY_MEV:.4f} MeV")
    print(
        f"After the collision the electron keeps {exit_speed_fraction:g} of its "
        f"speed and leaves at {exit_angle_deg:g} deg from the beam"
    )
    print()
    print(
        f"{'T':>8} {'beta':>10} {'gamma':>10} {'theta_max':>10} {'acos(beta)':>11} "
        f"{'1/gamma':>9} {'forward':>8} {'50% cone':>9}"
    )
    print(
        f"{'':>8} {'':>10} {'':>10} {'(deg)':>10} {'(deg)':>11} "
        f"{'(deg)':>9} {'share':>8} {'(deg)':>9}"
    )

    # Collect per-energy results for the plots
    results = []
    for t_mev in kinetic_energies_mev:
        beta, gamma = speed_from_kinetic_energy(t_mev)
        beta_in = np.array([beta, 0.0, 0.0])
        beta_out = (
            exit_speed_fraction
            * beta
            * np.array([np.cos(exit_angle), np.sin(exit_angle), 0.0])
        )
        theta_max = peak_angle(beta_in, beta_out)
        forward, half_angle = energy_cone(beta_in, beta_out)
        results.append((t_mev, beta, gamma, beta_in, beta_out, theta_max, half_angle))
        print(
            f"{energy_label(t_mev):>8} {beta:10.6f} {gamma:10.3f} "
            f"{np.degrees(theta_max):10.3f} {np.degrees(np.arccos(beta)):11.3f} "
            f"{np.degrees(1 / gamma):9.3f} {forward:8.4f} {np.degrees(half_angle):9.3f}"
        )
    print()
    print("acos(beta) is the exact peak angle for an electron that stops dead;")
    print("1/gamma is its small-angle (high-energy) approximation.")

    fig, axes = plt.subplots(
        2, 3, num=Path(__file__).name, figsize=(16, 10), layout="constrained"
    )
    fig.suptitle(
        "Bremsstrahlung from a LINAC electron striking a target: "
        "the radiation cone narrows as the energy rises"
    )
    map_axes = axes.ravel()[:5]
    curve_ax = axes[1, 2]

    # Grid of points in the plane that holds the beam (the x axis) and the
    # electron's exit path. Each point (x, y) sees the radiation leaving the
    # collision at angle theta from +x, with sign because a deflected electron
    # makes a pattern that is not symmetric about the beam.
    extent = 1.0
    n_points = 601
    x, y = np.meshgrid(
        np.linspace(-extent, extent, n_points),
        np.linspace(-extent, extent, n_points),
    )
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    # Far field intensity = (energy per solid angle) / r^2. Hide the region
    # very close to the collision where 1/r^2 blows up. One log color scale
    # serves every map so the panels can be compared directly. Six decades
    # are needed because at high energy the faint halo around the narrow
    # cone is a million times dimmer than the cone itself.
    r_min = 0.04
    norm = LogNorm(vmin=1e-6 / r_min**2, vmax=1 / r_min**2)
    phi = np.linspace(-np.pi, np.pi, 2_000)

    for ax, (t_mev, beta, gamma, beta_in, beta_out, theta_max, _) in zip(
        map_axes, results
    ):
        peak = in_plane_pattern(np.array(theta_max), beta_in, beta_out)
        pattern = in_plane_pattern(theta, beta_in, beta_out) / peak
        intensity = pattern / np.maximum(r, r_min) ** 2
        intensity = np.ma.masked_where(r < r_min, np.clip(intensity, norm.vmin, None))
        ax.imshow(
            intensity,
            cmap="inferno",
            norm=norm,
            origin="lower",
            extent=(-extent, extent, -extent, extent),
        )

        # Overlay the angular pattern itself as a polar curve
        lobe = 0.9 * extent * in_plane_pattern(phi, beta_in, beta_out) / peak
        ax.plot(lobe * np.cos(phi), lobe * np.sin(phi), color="cyan", lw=1.2)

        # Mark the direction of peak emission (and its mirror image when the
        # pattern is symmetric about the beam)
        mirrors = (1, -1) if exit_angle_deg == 0 else (1,)
        for sign in mirrors:
            ax.plot(
                [0, extent * np.cos(theta_max)],
                [0, sign * extent * np.sin(theta_max)],
                color="white",
                ls="--",
                lw=1,
            )

        # Draw the thin target foil and the incoming electron
        ax.add_patch(Rectangle((-0.02, -0.75), 0.04, 1.5, color="lightgray", zorder=3))
        ax.text(0.05, 0.72, "target", color="white", fontsize=9)
        ax.annotate(
            "",
            xy=(-0.06, 0),
            xytext=(-0.85, 0),
            arrowprops={"arrowstyle": "->", "color": "deepskyblue", "lw": 2},
            zorder=4,
        )
        ax.text(-0.85, 0.06, r"e$^-$ from LINAC", color="deepskyblue", fontsize=10)

        # Draw the electron leaving the collision, if it survives
        if exit_speed_fraction > 0:
            ax.annotate(
                "",
                xy=(
                    0.4 * exit_speed_fraction * np.cos(exit_angle),
                    0.4 * exit_speed_fraction * np.sin(exit_angle),
                ),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "->", "color": "deepskyblue", "lw": 2},
                zorder=4,
            )

        ax.set_title(
            f"T = {energy_label(t_mev)}\n"
            rf"$\beta$ = {beta:.6f},  $\gamma$ = {gamma:.3f},  "
            rf"$\theta_{{max}}$ = {np.degrees(theta_max):.2f}$^\circ$"
        )
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xlabel("x  (beam direction)")
        ax.set_ylabel("y")

    # Every map shares the same norm and colormap, so one colorbar built from
    # that pair describes all of them
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="inferno"),
        ax=list(map_axes),
        location="bottom",
        shrink=0.4,
        label=r"Relative intensity  (d$I$/d$\Omega$) / $r^2$  (same scale in every map)",
    )

    # Sixth panel: how the cone angle shrinks with energy
    sweep_mev = np.logspace(-3, 3, 300)
    sweep_peak = []
    sweep_beta = []
    sweep_gamma = []
    for t_mev in sweep_mev:
        beta, gamma = speed_from_kinetic_energy(t_mev)
        beta_in = np.array([beta, 0.0, 0.0])
        beta_out = (
            exit_speed_fraction
            * beta
            * np.array([np.cos(exit_angle), np.sin(exit_angle), 0.0])
        )
        sweep_peak.append(peak_angle(beta_in, beta_out))
        sweep_beta.append(beta)
        sweep_gamma.append(gamma)
    sweep_peak = np.degrees(sweep_peak)
    sweep_beta = np.array(sweep_beta)
    sweep_gamma = np.array(sweep_gamma)

    curve_ax.loglog(sweep_mev, sweep_peak, color="C0", lw=2, label=r"$\theta_{max}$")
    curve_ax.loglog(
        sweep_mev,
        np.degrees(np.arccos(sweep_beta)),
        color="C3",
        ls=":",
        lw=2,
        label=r"$\arccos\beta$ (electron stops dead)",
    )
    curve_ax.loglog(
        sweep_mev,
        np.degrees(1 / sweep_gamma),
        color="k",
        ls="--",
        lw=1,
        label=r"$1/\gamma$",
    )
    chosen_mev = [res[0] for res in results]
    curve_ax.plot(
        chosen_mev,
        [np.degrees(res[5]) for res in results],
        "o",
        color="C0",
        ms=7,
        label=r"$\theta_{max}$ of the maps",
    )
    curve_ax.plot(
        chosen_mev,
        [np.degrees(res[6]) for res in results],
        "s",
        color="C1",
        ms=7,
        label="half-angle holding 50% of the energy",
    )
    curve_ax.axhline(90, color="gray", lw=0.8)
    curve_ax.text(2.0, 84, r"90$^\circ$: sideways doughnut", color="gray", va="top")
    curve_ax.set_title("Cone angle versus electron kinetic energy")
    curve_ax.set_xlabel("Kinetic energy T (MeV)")
    curve_ax.set_ylabel("Angle from the beam (degrees)")
    curve_ax.grid(True, which="both", alpha=0.3)
    curve_ax.legend(loc="lower left", fontsize=9)

    plt.show()


if __name__ == "__main__":
    main()
