#!/usr/bin/env -S uv run
"""electron_radiation.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# An electron moves along the +x axis and is accelerated along that same line,
# as in a linear accelerator (LINAC). The Lienard result gives the power it
# radiates per unit solid angle, at angle theta measured from the velocity:
#
#   dP/dOmega = q^2 a^2 / (16 pi^2 eps0 c^3) * sin^2(theta) / (1 - beta cos(theta))^5
#
# The constant in front only sets the overall scale, so the plot shows the
# shape of the pattern normalized to its peak value.


def angular_distribution(theta: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute the Lienard angular distribution for acceleration parallel to velocity.

    Parameters
    ----------
    theta : np.ndarray
        Angle from the direction of motion, in radians.
    beta : float
        Electron speed as a fraction of the speed of light (0 <= beta < 1).

    Returns
    -------
    np.ndarray
        dP/dOmega normalized so that its maximum value is 1.
    """
    pattern = np.sin(theta) ** 2 / (1 - beta * np.cos(theta)) ** 5
    return pattern / peak_value(beta)


def peak_angle(beta: float) -> float:
    """
    Return the angle (radians) where the emitted power per solid angle peaks.

    Setting d/dtheta of sin^2(theta) / (1 - beta cos(theta))^5 to zero gives
    cos(theta_max) = (sqrt(1 + 15 beta^2) - 1) / (3 beta).
    """
    if beta == 0:
        # Nonrelativistic Larmor doughnut peaks perpendicular to the motion
        return np.pi / 2
    return float(np.arccos((np.sqrt(1 + 15 * beta**2) - 1) / (3 * beta)))


def peak_value(beta: float) -> float:
    """Return the unnormalized dP/dOmega at its peak angle."""
    theta_max = peak_angle(beta)
    return float(np.sin(theta_max) ** 2 / (1 - beta * np.cos(theta_max)) ** 5)


def main():
    # TODO: Change the electron speed (fraction of c, 0 <= beta < 1)
    beta = 0.5

    gamma = 1 / np.sqrt(1 - beta**2)
    theta_max = peak_angle(beta)

    # Forward and backward lobes of the pattern
    theta_fwd = np.radians(30)
    theta_bwd = np.radians(150)
    fwd_bwd_ratio = angular_distribution(
        np.array(theta_fwd), beta
    ) / angular_distribution(np.array(theta_bwd), beta)

    print(f"Electron speed:          beta  = {beta:.4f}")
    print(f"Lorentz factor:          gamma = {gamma:.4f}")
    print(f"Peak emission angle:     theta_max = {np.degrees(theta_max):.3f} deg")
    print(
        f"Ultra-relativistic est.: 1/(2 gamma) = {np.degrees(1 / (2 * gamma)):.3f} deg"
    )
    print(f"Power ratio at 30 deg vs 150 deg: {float(fwd_bwd_ratio):,.3f}")

    # Grid of points in a plane that contains the line of motion (the x axis).
    # Every point (x, y) sees the radiation leaving at angle theta from +x.
    extent = 1.0
    n_points = 801
    x, y = np.meshgrid(
        np.linspace(-extent, extent, n_points),
        np.linspace(-extent, extent, n_points),
    )
    r = np.hypot(x, y)
    theta = np.arctan2(np.abs(y), x)  # pattern is symmetric about the x axis

    # Far field intensity = (power per solid angle) / r^2.
    # Hide the region very close to the electron where 1/r^2 blows up.
    r_min = 0.04
    intensity = angular_distribution(theta, beta) / np.maximum(r, r_min) ** 2
    intensity = np.ma.masked_where(r < r_min, intensity)

    # A log color scale shows both the bright lobes and the dark axis
    i_max = intensity.max()
    norm = LogNorm(vmin=i_max * 1e-4, vmax=i_max)
    intensity = np.clip(intensity, norm.vmin, None)

    plt.figure(Path(__file__).name, figsize=(9, 7.5))
    mesh = plt.pcolormesh(x, y, intensity, cmap="inferno", norm=norm, shading="auto")
    plt.colorbar(mesh, label="Relative intensity  (dP/d$\\Omega$) / $r^2$")

    # Overlay the angular pattern itself as a polar curve
    phi = np.linspace(-np.pi, np.pi, 2_000)
    lobe = 0.9 * extent * angular_distribution(np.abs(phi), beta)
    plt.plot(
        lobe * np.cos(phi),
        lobe * np.sin(phi),
        color="cyan",
        lw=1.5,
        label=r"d$P$/d$\Omega$ (polar plot)",
    )

    # Mark the directions of peak emission
    for sign in (1, -1):
        plt.plot(
            [0, extent * np.cos(theta_max)],
            [0, sign * extent * np.sin(theta_max)],
            color="white",
            ls="--",
            lw=1,
            label=rf"$\theta_{{max}}$ = {np.degrees(theta_max):.1f}$^\circ$"
            if sign == 1
            else None,
        )

    # Draw the electron and its velocity (and acceleration) direction
    plt.plot(0, 0, "o", color="deepskyblue", ms=8)
    plt.annotate(
        "",
        xy=(0.3, 0),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": "deepskyblue", "lw": 2},
    )
    plt.text(0.12, -0.08, r"$\vec{v},\ \vec{a}$", color="deepskyblue", fontsize=12)

    plt.title(
        "Radiation from an electron accelerated along a straight line\n"
        rf"$\beta$ = {beta:.3f},  $\gamma$ = {gamma:.3f}"
    )
    plt.xlabel("x  (direction of motion)")
    plt.ylabel("y")
    plt.xlim(-extent, extent)
    plt.ylim(-extent, extent)
    plt.gca().set_aspect("equal")
    plt.legend(loc="lower left", facecolor="black", labelcolor="white")
    plt.show()


if __name__ == "__main__":
    main()
