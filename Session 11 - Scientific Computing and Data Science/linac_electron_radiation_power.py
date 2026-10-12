#!/usr/bin/env -S uv run
"""linac_electron_radiation_power.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, physical_constants

# An electron in a linear accelerator (LINAC) is pushed along a straight line,
# so its acceleration is parallel to its velocity. The Lienard result gives
# the total power it radiates in two equivalent forms:
#
#   (1)  P = q^2 a^2 gamma^6 / (6 pi eps0 c^3)
#   (2)  P = q^2 (dp/dt)^2 / (6 pi eps0 m^2 c^3)
#
# Form (1) suggests a huge gamma^6 growth with energy. Form (2) shows that, for
# a fixed push dp/dt, the radiated power does not depend on energy at all.
# They agree because, for straight-line motion, dp/dt = gamma^3 m a.
#
# A LINAC applies a nearly constant accelerating field, so dp/dt = q E is
# fixed. This script follows an electron as it gains energy at that fixed
# gradient and evaluates both forms side by side.


def power_from_acceleration(a: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Radiated power from form (1), using the acceleration a = dv/dt.

    Parameters
    ----------
    a : np.ndarray
        Acceleration in m/s^2 (parallel to the velocity).
    gamma : np.ndarray
        Lorentz factor of the electron.

    Returns
    -------
    np.ndarray
        Radiated power in watts.
    """
    return e**2 * a**2 * gamma**6 / (6 * np.pi * epsilon_0 * c**3)


def power_from_force(dp_dt: np.ndarray) -> np.ndarray:
    """
    Radiated power from form (2), using the force dp/dt.

    Parameters
    ----------
    dp_dt : np.ndarray
        Rate of change of momentum (force) in newtons.

    Returns
    -------
    np.ndarray
        Radiated power in watts.
    """
    return e**2 * dp_dt**2 / (6 * np.pi * epsilon_0 * m_e**2 * c**3)


def main():
    # TODO: Change the LINAC accelerating gradient (MeV per meter)
    gradient_mev_per_m = 20.0

    # Electron rest energy and classical electron radius
    rest_energy_mev = m_e * c**2 / (1e6 * e)
    r_e = physical_constants["classical electron radius"][0]

    # Constant accelerating field E, so the force is F = qE for every electron
    field = gradient_mev_per_m * 1e6  # volts per meter
    force = e * field  # newtons, equal to dp/dt

    # Range of total electron energies, from just above rest to 10 GeV
    total_energy_mev = np.logspace(np.log10(rest_energy_mev * 1.001), 4, 400)
    gamma = total_energy_mev / rest_energy_mev
    beta = np.sqrt(1 - 1 / gamma**2)

    # For straight-line motion dp/dt = gamma^3 m a, so the same force produces
    # less and less acceleration as the electron approaches the speed of light
    accel = force / (gamma**3 * m_e)
    dp_dt = np.full_like(gamma, force)

    p_form1 = power_from_acceleration(accel, gamma)
    p_form2 = power_from_force(dp_dt)

    # Rate of energy gain dE/dt = F v, and the fraction lost to radiation
    energy_gain_rate = force * beta * c  # watts
    loss_fraction = p_form1 / energy_gain_rate

    # Ultra-relativistic estimate: (2/3) (r_e / m c^2) (dE/dx)
    loss_estimate = (2 / 3) * r_e * gradient_mev_per_m / rest_energy_mev

    print(f"Accelerating gradient:       {gradient_mev_per_m:.1f} MeV/m")
    print(f"Force on electron (dp/dt):   {force:.4e} N")
    print()
    print(
        f"{'E (MeV)':>10} {'gamma':>10} {'a (m/s^2)':>12} "
        f"{'P form 1 (W)':>14} {'P form 2 (W)':>14} {'P_rad/(dE/dt)':>14}"
    )
    for target_mev in (0.6, 1, 10, 100, 1000, 10_000):
        i = int(np.argmin(np.abs(total_energy_mev - target_mev)))
        print(
            f"{total_energy_mev[i]:10.3f} {gamma[i]:10.3f} {accel[i]:12.4e} "
            f"{p_form1[i]:14.6e} {p_form2[i]:14.6e} {loss_fraction[i]:14.4e}"
        )
    print()
    max_rel_diff = np.max(np.abs(p_form1 - p_form2) / p_form2)
    print(f"Largest relative difference between forms: {max_rel_diff:.2e}")
    print(
        f"Ultra-relativistic loss estimate (2/3)(r_e/mc^2)(dE/dx) = {loss_estimate:.4e}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, num=Path(__file__).name, figsize=(13, 5.5))
    fig.suptitle(
        "Radiation from an electron in a LINAC at a fixed gradient of "
        f"{gradient_mev_per_m:g} MeV/m"
    )

    # Left: the two factors inside form (1) move in opposite directions
    ax1.loglog(gamma, accel**2 / accel[0] ** 2, label=r"$a^2$ (relative)")
    ax1.loglog(gamma, gamma**6 / gamma[0] ** 6, label=r"$\gamma^6$ (relative)")
    ax1.loglog(
        gamma,
        (accel**2 * gamma**6) / (accel[0] ** 2 * gamma[0] ** 6),
        "k--",
        lw=2,
        label=r"$a^2\gamma^6$ (product)",
    )
    ax1.set_title(r"Form 1: $a^2$ falls as $\gamma^{-6}$, cancelling $\gamma^6$")
    ax1.set_xlabel(r"Lorentz factor $\gamma$")
    ax1.set_ylabel("Value relative to the lowest energy")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Right: both forms give the same flat radiated power
    ax2.semilogx(
        total_energy_mev,
        p_form1,
        lw=5,
        alpha=0.5,
        label=r"Form 1: $q^2 a^2 \gamma^6 / (6\pi\varepsilon_0 c^3)$",
    )
    ax2.semilogx(
        total_energy_mev,
        p_form2,
        "k--",
        label=r"Form 2: $q^2 (dp/dt)^2 / (6\pi\varepsilon_0 m^2 c^3)$",
    )
    ax2.set_title("Radiated power: both forms agree and stay constant")
    ax2.set_xlabel("Electron total energy (MeV)")
    ax2.set_ylabel("Radiated power P (W)")
    ax2.set_ylim(0, 1.5 * p_form2[0])
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="lower right")
    ax2.text(
        0.03,
        0.95,
        f"P_rad / (dE/dt) at high energy = {loss_fraction[-1]:.2e}",
        transform=ax2.transAxes,
        va="top",
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
