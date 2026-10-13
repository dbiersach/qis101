#!/usr/bin/env -S uv run
"""linac_electron_bremsstrahlung_power.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import N_A, alpha, c, e, m_e, physical_constants
from scipy.integrate import cumulative_trapezoid

# An electron that enters a target loses energy in two ways. Collisions with
# atomic electrons ionize and excite the atoms, and that energy ends up as
# heat. Close passes by nuclei produce bremsstrahlung photons. The stopping
# power S = -dT/dx measures each loss per unit path length, and the ratio of
# the two decides how much of the beam energy leaves the target as X-rays.
#
# Collision stopping power (Bethe formula for electrons):
#
#   S_col / rho = (K/2) (Z/A) / beta^2 * [ ln( tau^2 (tau+2) / (2 (I/mc^2)^2) ) + F(tau) - delta ]
#   F(tau) = 1 - beta^2 + [ tau^2 / 8 - (2 tau + 1) ln 2 ] / (tau + 1)^2,   tau = T / mc^2
#
# Radiative stopping power at high energy (Bethe-Heitler, complete screening):
#
#   S_rad / rho = E / X0,   E = T + mc^2,   X0 = radiation length
#
# Radiation yield, the fraction of an electron's kinetic energy T0 that turns
# into photons as it slows to rest in a thick target:
#
#   Y(T0) = (1/T0) * Integral from 0 to T0 of  S_rad / (S_col + S_rad)  dT
#
# The script computes the Bethe formula and the radiation length from
# scratch, reads the NIST ESTAR stopping-power tables for the full energy
# range, integrates them to get the yield, and checks the result against the
# yield column of the same tables.

# Electron rest energy in MeV and classical electron radius in cm
REST_ENERGY_MEV = m_e * c**2 / (1e6 * e)
R_E_CM = physical_constants["classical electron radius"][0] * 100

# Bethe constant K = 4 pi N_A r_e^2 m c^2, about 0.3071 MeV cm^2 / mol
K_BETHE = 4 * np.pi * N_A * R_E_CM**2 * REST_ENERGY_MEV

# Tungsten: atomic data plus the Sternheimer density-effect parameters,
# all from the Particle Data Group tables
TUNGSTEN = {
    "Z": 74,
    "A": 183.84,
    "I_ev": 727.0,
    "C": 5.4059,
    "x0": 0.2167,
    "x1": 3.4960,
    "a": 0.1551,
    "k": 2.8447,
    "delta0": 0.14,
}


def read_estar(
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an ESTAR table saved as a comma-separated file.

    Parameters
    ----------
    file_path : Path
        Full path of the file. Lines starting with # are comments.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Kinetic energy (MeV), collision stopping power (MeV cm^2/g),
        radiative stopping power (MeV cm^2/g), and NIST's radiation yield.
    """
    kinetic, s_col, s_rad, yield_nist = np.genfromtxt(
        file_path, delimiter=",", comments="#", unpack=True
    )
    return kinetic, s_col, s_rad, yield_nist


def density_effect(beta_gamma: np.ndarray, material: dict) -> np.ndarray:
    """
    Sternheimer's density-effect correction delta for the Bethe formula.

    Parameters
    ----------
    beta_gamma : np.ndarray
        Momentum divided by m c, which is beta * gamma.
    material : dict
        Holds the parameters C, x0, x1, a, k, and delta0.

    Returns
    -------
    np.ndarray
        The correction delta at each energy.
    """
    x = np.log10(beta_gamma)
    high = 2 * np.log(10) * x - material["C"]
    middle = high + material["a"] * (material["x1"] - x) ** material["k"]
    low = material["delta0"] * 10 ** (2 * (x - material["x0"]))
    return np.where(
        x >= material["x1"], high, np.where(x >= material["x0"], middle, low)
    )


def bethe_collision(kinetic_mev: np.ndarray, material: dict) -> np.ndarray:
    """
    Collision stopping power of electrons from the Bethe formula.

    Parameters
    ----------
    kinetic_mev : np.ndarray
        Kinetic energy in MeV.
    material : dict
        Holds Z, A, the mean excitation energy I_ev, and the density-effect
        parameters.

    Returns
    -------
    np.ndarray
        Mass collision stopping power in MeV cm^2 / g.
    """
    tau = kinetic_mev / REST_ENERGY_MEV
    gamma = tau + 1
    beta_sq = 1 - 1 / gamma**2
    i_ratio = material["I_ev"] * 1e-6 / REST_ENERGY_MEV
    log_term = np.log(tau**2 * (tau + 2) / (2 * i_ratio**2))
    f_term = 1 - beta_sq + (tau**2 / 8 - (2 * tau + 1) * np.log(2)) / (tau + 1) ** 2
    delta = density_effect(np.sqrt(beta_sq) * gamma, material)
    prefactor = (K_BETHE / 2) * (material["Z"] / material["A"]) / beta_sq
    return prefactor * (log_term + f_term - delta)


def radiation_length(z: int, a: float) -> float:
    """
    Radiation length X0 in g/cm^2 from Tsai's formula.

    Parameters
    ----------
    z : int
        Atomic number.
    a : float
        Atomic mass in g/mol.

    Returns
    -------
    float
        X0 in g/cm^2. An electron loses all but 1/e of its energy to
        bremsstrahlung in one radiation length, at high energy.
    """
    az = alpha * z
    coulomb = az**2 * (
        1 / (1 + az**2) + 0.20206 - 0.0369 * az**2 + 0.0083 * az**4 - 0.002 * az**6
    )
    l_rad = np.log(184.15 * z ** (-1 / 3))
    l_rad_prime = np.log(1194 * z ** (-2 / 3))
    inverse = (
        4 * alpha * R_E_CM**2 * (N_A / a) * (z**2 * (l_rad - coulomb) + z * l_rad_prime)
    )
    return float(1 / inverse)


def radiation_yield(
    kinetic: np.ndarray, s_col: np.ndarray, s_rad: np.ndarray
) -> np.ndarray:
    """
    Fraction of the kinetic energy radiated as an electron stops in a thick target.

    Parameters
    ----------
    kinetic : np.ndarray
        Kinetic energies in MeV, increasing.
    s_col, s_rad : np.ndarray
        Collision and radiative stopping powers at those energies.

    Returns
    -------
    np.ndarray
        Y(T0) at each tabulated energy T0.
    """
    share = s_rad / (s_col + s_rad)
    # Below the first tabulated energy the radiative share falls roughly in
    # proportion to T, so that stretch contributes a triangle of area
    # share[0] * kinetic[0] / 2
    integral = (
        cumulative_trapezoid(share, kinetic, initial=0) + 0.5 * share[0] * kinetic[0]
    )
    return integral / kinetic


def crossover_energy(
    kinetic: np.ndarray, s_col: np.ndarray, s_rad: np.ndarray
) -> float:
    """Return the kinetic energy (MeV) at which radiative loss first equals collision loss."""
    log_ratio = np.log(s_rad / s_col)
    i = int(np.argmax(log_ratio >= 0))
    # Straight-line interpolation in log-log between the two bracketing points
    frac = -log_ratio[i - 1] / (log_ratio[i] - log_ratio[i - 1])
    return float(
        np.exp(np.log(kinetic[i - 1]) + frac * np.log(kinetic[i] / kinetic[i - 1]))
    )


def main():
    # TODO: Change the target materials to compare. Each entry names an ESTAR
    # data file that sits in the same folder as this script
    materials = {
        "Tungsten": "estar_tungsten.csv",
        "Copper": "estar_copper.csv",
        "Water": "estar_water.csv",
    }

    # Read every table once. Each entry becomes (T, S_col, S_rad, Y_nist, Y_ours)
    tables = {}
    for name, file_name in materials.items():
        file_path = Path(__file__).parent / file_name
        kinetic, s_col, s_rad, yield_nist = read_estar(file_path)
        tables[name] = (
            kinetic,
            s_col,
            s_rad,
            yield_nist,
            radiation_yield(kinetic, s_col, s_rad),
        )
        print(f"Read {len(kinetic)} rows for {name.lower()} from {file_path}")
    print()

    # Theory for the tungsten target, computed from scratch
    kinetic_w, s_col_w, s_rad_w, yield_nist_w, yield_w = tables["Tungsten"]
    x0 = radiation_length(TUNGSTEN["Z"], TUNGSTEN["A"])
    fine_kinetic = np.logspace(-2, 3, 500)
    bethe_w = bethe_collision(fine_kinetic, TUNGSTEN)
    asymptote_w = (fine_kinetic + REST_ENERGY_MEV) / x0

    print(f"Radiation length of tungsten from Tsai's formula: X0 = {x0:.3f} g/cm^2")
    print(
        f"  (Particle Data Group value: 6.76 g/cm^2, or {6.76 / 19.30:.3f} cm at 19.30 g/cm^3)"
    )
    print()
    print(
        "Tungsten target. Stopping powers in MeV cm^2/g, from NIST ESTAR unless marked"
    )
    print(
        f"{'T (MeV)':>9} {'S_col':>8} {'Bethe':>8} {'S_rad':>9} {'E/X0':>9} "
        f"{'S_rad/S_col':>11} {'Y ours':>9} {'Y NIST':>9}"
    )
    for target_mev in (0.01, 0.1, 1, 6, 10, 100, 1000):
        i = int(np.argmin(np.abs(kinetic_w - target_mev)))
        bethe_here = float(bethe_collision(np.array(kinetic_w[i]), TUNGSTEN))
        asymptote_here = (kinetic_w[i] + REST_ENERGY_MEV) / x0
        print(
            f"{kinetic_w[i]:9.3g} {s_col_w[i]:8.3f} {bethe_here:8.3f} {s_rad_w[i]:9.4f} "
            f"{asymptote_here:9.4f} {s_rad_w[i] / s_col_w[i]:11.4f} "
            f"{yield_w[i]:9.4f} {yield_nist_w[i]:9.4f}"
        )
    print()

    for name, (kinetic, s_col, s_rad, yield_nist, yield_ours) in tables.items():
        worst = np.max(np.abs(yield_ours - yield_nist) / yield_nist)
        print(
            f"{name:>8}: S_rad = S_col at T = {crossover_energy(kinetic, s_col, s_rad):6.1f} MeV, "
            f"largest yield mismatch against NIST = {worst:.2%}"
        )
    print()

    # Compare with the loss inside the LINAC, from linac_electron_radiation_power.py:
    # (2/3) (r_e / m c^2) (dE/dx) at a gradient of 20 MeV/m
    gradient_mev_per_m = 20.0
    linac_loss = (2 / 3) * (R_E_CM / 100) * gradient_mev_per_m / REST_ENERGY_MEV
    i10 = int(np.argmin(np.abs(kinetic_w - 10)))
    print(
        f"Fraction of energy radiated inside the LINAC at {gradient_mev_per_m:g} MeV/m: {linac_loss:.2e}"
    )
    print(
        f"Fraction radiated by a 10 MeV electron stopping in tungsten:       {yield_w[i10]:.2f}"
    )
    print(f"Ratio: {yield_w[i10] / linac_loss:.1e}")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, num=Path(__file__).name, figsize=(17, 5.5), layout="constrained"
    )
    fig.suptitle(
        "How much of a LINAC electron's energy becomes bremsstrahlung in the target?"
    )

    # Left: the two stopping powers in tungsten, with the textbook formulas on top
    ax1.loglog(kinetic_w, s_col_w, color="C0", lw=2.5, label=r"$S_{col}$ (NIST ESTAR)")
    ax1.loglog(kinetic_w, s_rad_w, color="C3", lw=2.5, label=r"$S_{rad}$ (NIST ESTAR)")
    ax1.loglog(fine_kinetic, bethe_w, "k--", lw=1.2, label="Bethe formula")
    ax1.loglog(
        fine_kinetic, asymptote_w, "k:", lw=1.2, label=r"$E / X_0$ (complete screening)"
    )
    t_cross = crossover_energy(kinetic_w, s_col_w, s_rad_w)
    ax1.axvline(t_cross, color="gray", lw=0.8)
    ax1.text(t_cross * 1.15, 0.03, f"equal at\n{t_cross:.1f} MeV", color="gray")
    ax1.set_title("Tungsten: collision versus radiative loss")
    ax1.set_xlabel("Kinetic energy T (MeV)")
    ax1.set_ylabel(r"Mass stopping power $-dT/d(\rho x)$  (MeV cm$^2$/g)")
    ax1.set_ylim(1e-2, 1e3)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left")

    # Middle: the ratio for every material, crossing 1 at the critical energy
    for name, (kinetic, s_col, s_rad, _, _) in tables.items():
        ratio = s_rad / s_col
        t_cross = crossover_energy(kinetic, s_col, s_rad)
        (line,) = ax2.loglog(
            kinetic, ratio, lw=2, label=f"{name} (equal at {t_cross:.1f} MeV)"
        )
        ax2.plot(t_cross, 1, "o", color=line.get_color())
    ax2.axhline(1, color="gray", lw=0.8)
    ax2.set_title(r"Ratio $S_{rad} / S_{col}$: radiation wins above the crossing")
    ax2.set_xlabel("Kinetic energy T (MeV)")
    ax2.set_ylabel(r"$S_{rad} / S_{col}$")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left")

    # Right: the radiation yield, integrated here and checked against NIST
    for name, (kinetic, _, _, yield_nist, yield_ours) in tables.items():
        (line,) = ax3.semilogx(
            kinetic, yield_ours, lw=2, label=f"{name} (integrated here)"
        )
        ax3.semilogx(
            kinetic[::6],
            yield_nist[::6],
            "o",
            color=line.get_color(),
            mfc="none",
            ms=6,
            label=f"{name} (NIST yield column)",
        )
    # The labels sit in the empty region below the legend, with arrows to the
    # tungsten curve
    for t_mark, text, text_y in ((0.1, "X-ray tube", 0.16), (6, "medical LINAC", 0.42)):
        i = int(np.argmin(np.abs(kinetic_w - t_mark)))
        ax3.axvline(t_mark, color="gray", lw=0.8, ls=":")
        ax3.annotate(
            f"{text}\nW: {yield_w[i]:.1%}",
            xy=(t_mark, yield_w[i]),
            xytext=(0.013, text_y),
            arrowprops={"arrowstyle": "->", "color": "gray"},
            fontsize=9,
        )
    ax3.set_title("Radiation yield: share of T that leaves as photons")
    ax3.set_xlabel("Initial kinetic energy T (MeV)")
    ax3.set_ylabel("Radiation yield Y")
    ax3.set_ylim(0, 1)
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8)

    plt.show()


if __name__ == "__main__":
    main()
