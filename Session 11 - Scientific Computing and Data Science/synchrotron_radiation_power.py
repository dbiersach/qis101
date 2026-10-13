#!/usr/bin/env -S uv run
"""synchrotron_radiation_power.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.constants import c, e, epsilon_0, hbar, m_e, physical_constants
from scipy.integrate import cumulative_trapezoid, quad
from scipy.special import kv

# A charged particle on a circular orbit of radius rho is accelerated toward
# the center with a = beta^2 c^2 / rho, at right angles to its velocity. The
# Lienard result for the radiated power is
#
#   P = q^2 a^2 gamma^4 / (6 pi eps0 c^3) = q^2 c beta^4 gamma^4 / (6 pi eps0 rho^2)
#
# One turn takes 2 pi rho / (beta c), so the energy lost per turn is
#
#   U0 = q^2 beta^3 gamma^4 / (3 eps0 rho)
#
# The radiation is a flash each time the particle's headlight sweeps past an
# observer, and the spectrum of a flash extends up to the critical photon energy
#
#   eps_c = (3/2) hbar c gamma^3 / rho
#
# with half the power above eps_c and half below. The shape of the spectrum
# is a universal function of x = eps / eps_c,
#
#   S(x) = (9 sqrt(3) / (8 pi)) x Integral from x to infinity of K_{5/3}(t) dt
#
# normalized so that its integral over x is 1.

# Electron rest energy in MeV and classical electron radius in m
ELECTRON_MASS_MEV = m_e * c**2 / (1e6 * e)
R_E = physical_constants["classical electron radius"][0]


def lorentz_factors(
    energy_gev: np.ndarray, mass_mev: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gamma, beta) for a particle of total energy energy_gev and rest energy mass_mev."""
    gamma = energy_gev * 1e3 / mass_mev
    beta = np.sqrt(1 - 1 / gamma**2)
    return gamma, beta


def energy_loss_per_turn(
    energy_gev: np.ndarray, radius_m: np.ndarray, mass_mev: float
) -> np.ndarray:
    """
    Energy radiated in one revolution, in eV.

    Parameters
    ----------
    energy_gev : np.ndarray
        Total energy of the particle in GeV.
    radius_m : np.ndarray
        Bending radius of the orbit in meters.
    mass_mev : float
        Rest energy of the particle in MeV.

    Returns
    -------
    np.ndarray
        U0 = q^2 beta^3 gamma^4 / (3 eps0 rho), converted from joules to eV.
    """
    gamma, beta = lorentz_factors(energy_gev, mass_mev)
    u0_joule = e**2 * beta**3 * gamma**4 / (3 * epsilon_0 * radius_m)
    return u0_joule / e


def critical_photon_energy(
    energy_gev: np.ndarray, radius_m: np.ndarray, mass_mev: float
) -> np.ndarray:
    """
    Critical photon energy of the synchrotron spectrum, in eV.

    Parameters
    ----------
    energy_gev : np.ndarray
        Total energy of the particle in GeV.
    radius_m : np.ndarray
        Bending radius of the orbit in meters.
    mass_mev : float
        Rest energy of the particle in MeV.

    Returns
    -------
    np.ndarray
        eps_c = (3/2) hbar c gamma^3 / rho, converted from joules to eV.
    """
    gamma, _ = lorentz_factors(energy_gev, mass_mev)
    return 1.5 * hbar * c * gamma**3 / radius_m / e


def synchrotron_spectrum(x: np.ndarray) -> np.ndarray:
    """
    Universal synchrotron spectrum S(x), the power per unit x = eps / eps_c.

    Parameters
    ----------
    x : np.ndarray
        Photon energy divided by the critical energy.

    Returns
    -------
    np.ndarray
        S(x) = (9 sqrt(3) / (8 pi)) x Integral_x^inf K_{5/3}(t) dt, which
        integrates to 1 over all x.
    """
    tail = np.array([quad(lambda t: kv(5 / 3, t), xi, np.inf)[0] for xi in x])
    return 9 * np.sqrt(3) / (8 * np.pi) * x * tail


def format_ev(value_ev: float) -> str:
    """Format an energy in eV with the most readable prefix."""
    for scale, unit in ((1e9, "GeV"), (1e6, "MeV"), (1e3, "keV")):
        if value_ev >= scale:
            return f"{value_ev / scale:.3g} {unit}"
    return f"{value_ev:.3g} eV"


def main():
    # TODO: Change the particle. Electrons are 0.511 MeV, protons 938.27 MeV
    particle_mass_mev = ELECTRON_MASS_MEV

    # TODO: Change the machines drawn on the maps: (beam energy GeV, bending radius m, beam current A)
    machines = {
        "NSLS-II": (3.0, 25.0, 0.5),
        "APS": (7.0, 38.96, 0.1),
        "SPring-8": (8.0, 39.27, 0.1),
        "FCC-ee (Z)": (45.6, 10760.0, 1.39),
        "LEP2": (104.5, 3026.0, 0.006),
        "FCC-ee (tt)": (182.5, 10760.0, 0.0054),
    }

    # Where each machine's name goes on the maps, so that neighbors do not
    # overlap: (x offset, y offset) in points and the horizontal alignment
    label_offsets = {
        "NSLS-II": (6, -12, "left"),
        "APS": (-8, -14, "right"),
        "SPring-8": (6, 4, "left"),
        "FCC-ee (Z)": (-8, 4, "right"),
        "FCC-ee (tt)": (-8, -12, "right"),
    }

    # The constant in the practical formula U0 = C_gamma E^4 / rho
    c_gamma = (4 * np.pi / 3) * R_E / (particle_mass_mev * 1e-3) ** 3  # m / GeV^3
    print(f"Particle rest energy: {particle_mass_mev:.3f} MeV")
    print(
        f"U0 = C_gamma E^4 / rho with C_gamma = {c_gamma:.4e} m/GeV^3, so for E in GeV and rho in m"
    )
    print(f"    U0 = {c_gamma * 1e9 / 1e3:.2f} keV x E^4 / rho")
    print()
    print(
        f"{'Machine':>12} {'E (GeV)':>8} {'rho (m)':>8} {'gamma':>10} {'U0 / turn':>11} "
        f"{'eps_c':>10} {'I (A)':>7} {'P radiated':>11}"
    )
    for name, (energy, radius, current) in machines.items():
        gamma, _ = lorentz_factors(energy, particle_mass_mev)
        u0 = energy_loss_per_turn(energy, radius, particle_mass_mev)
        eps_c = critical_photon_energy(energy, radius, particle_mass_mev)
        power_w = (
            u0 * current
        )  # eV per turn x (turns per second x charge) = U0[J] I / e
        print(
            f"{name:>12} {energy:8.1f} {radius:8.0f} {gamma:10.0f} {format_ev(u0):>11} "
            f"{format_ev(eps_c):>10} {current:7.3f} {power_w / 1e6:8.2f} MW"
        )
    print()

    # The LHC bends 7 TeV protons on the same 2804 m radius; compare with electrons there
    lhc_energy, lhc_radius = 7000.0, 2804.0
    proton_mass_mev = physical_constants["proton mass energy equivalent in MeV"][0]
    u0_proton = energy_loss_per_turn(lhc_energy, lhc_radius, proton_mass_mev)
    u0_electron = energy_loss_per_turn(lhc_energy, lhc_radius, ELECTRON_MASS_MEV)
    print(
        f"LHC protons at {lhc_energy:g} GeV on a {lhc_radius:g} m radius lose {format_ev(u0_proton)} per turn."
    )
    print(
        f"Electrons at the same energy and radius would lose {format_ev(u0_electron)} per turn,"
    )
    print(
        f"{u0_electron / u0_proton:.1e} times more, which is (m_p / m_e)^4 = {(proton_mass_mev / ELECTRON_MASS_MEV) ** 4:.1e}."
    )
    print()

    # The spectrum, and a check that half the power lies above the critical energy
    x = np.logspace(-3, 1.3, 400)
    spectrum = synchrotron_spectrum(x)
    cumulative = cumulative_trapezoid(spectrum, x, initial=0)
    print(f"Integral of S(x) from 0 to infinity: {cumulative[-1]:.4f}   (expected 1)")
    print(
        f"Fraction of the power below eps_c:   {np.interp(1.0, x, cumulative):.4f}   (expected 0.5)"
    )
    print(f"S(x) peaks at x = {x[np.argmax(spectrum)]:.3f}")

    # Maps of U0 and eps_c over beam energy and bending radius
    energy_grid = np.logspace(-2, 3, 400)  # 10 MeV to 1 TeV
    radius_grid = np.logspace(0, 5, 400)  # 1 m to 100 km
    energy_mesh, radius_mesh = np.meshgrid(energy_grid, radius_grid)
    u0_map = energy_loss_per_turn(energy_mesh, radius_mesh, particle_mass_mev)
    eps_map = critical_photon_energy(energy_mesh, radius_mesh, particle_mass_mev)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, num=Path(__file__).name, figsize=(18, 6), layout="constrained"
    )
    fig.suptitle(
        f"Synchrotron radiation from a {particle_mass_mev:.3g} MeV particle on a circular orbit: "
        "energy lost per turn, photon energy, and spectrum"
    )

    panels = (
        (
            ax1,
            u0_map,
            "Energy lost per turn  $U_0 \\propto E^4 / \\rho$",
            1.0,
            1e11,
            [1e3, 1e6, 1e9],
            {1e3: "1 keV", 1e6: "1 MeV", 1e9: "1 GeV"},
        ),
        (
            ax2,
            eps_map,
            "Critical photon energy  $\\varepsilon_c \\propto E^3 / \\rho$",
            1e-3,
            1e8,
            [1.0, 1e3, 1e5],
            {
                1.0: "1 eV (visible)",
                1e3: "1 keV (soft X-ray)",
                1e5: "100 keV (hard X-ray)",
            },
        ),
    )
    for ax, data, title, vmin, vmax, levels, names in panels:
        mesh = ax.pcolormesh(
            energy_mesh,
            radius_mesh,
            np.clip(data, vmin, vmax),
            cmap="inferno",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="auto",
        )
        contours = ax.contour(
            energy_mesh, radius_mesh, data, levels=levels, colors="white", linewidths=1
        )
        ax.clabel(contours, fmt=names, fontsize=9)
        for name, (energy, radius, _) in machines.items():
            ax.plot(energy, radius, "o", color="deepskyblue", ms=7, mec="white")
            dx, dy, align = label_offsets.get(name, (6, 4, "left"))
            ax.annotate(
                name,
                (energy, radius),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=align,
                color="white",
                fontsize=9,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Beam energy E (GeV)")
        ax.set_ylabel(r"Bending radius $\rho$ (m)")
        fig.colorbar(mesh, ax=ax, shrink=0.85, label="eV")

    # Right: the universal spectrum
    ax3.loglog(x, spectrum, color="C3", lw=2)
    ax3.fill_between(
        x, spectrum, 1e-4, where=x >= 1, color="C3", alpha=0.25, label="half the power"
    )
    ax3.axvline(1, color="gray", lw=0.8)
    ax3.text(1.1, 0.5, r"$\varepsilon = \varepsilon_c$", color="gray")
    ax3.set_xlim(1e-3, 10)
    ax3.set_ylim(1e-3, 1)
    ax3.set_title("Universal spectrum: power per unit photon energy")
    ax3.set_xlabel(
        r"Photon energy / critical energy,  $x = \varepsilon / \varepsilon_c$"
    )
    ax3.set_ylabel(r"$S(x)$")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="lower left")

    plt.show()


if __name__ == "__main__":
    main()
