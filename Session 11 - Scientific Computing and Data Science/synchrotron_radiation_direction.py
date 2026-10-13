#!/usr/bin/env -S uv run
"""synchrotron_radiation_direction.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.integrate import trapezoid

# An electron on a circular orbit is accelerated toward the center of the
# ring, at right angles to its velocity. The Lienard result for the power it
# radiates per unit solid angle, in the direction of the unit vector n, is
#
#   dP/dOmega = q^2 / (16 pi^2 eps0 c) * |n x ((n - beta) x beta_dot)|^2 / (1 - n . beta)^5
#
# With the velocity along +x and the acceleration along +y this becomes
#
#   dP/dOmega = q^2 a^2 / (16 pi^2 eps0 c^3) * 1 / (1 - beta cos(theta))^3
#               * [ 1 - sin^2(theta) cos^2(phi) / (gamma^2 (1 - beta cos(theta))^2) ]
#
# where theta is measured from the velocity and phi is the azimuth around it,
# measured from the acceleration. Integrated over all directions,
#
#   P = q^2 a^2 gamma^4 / (6 pi eps0 c^3)
#
# The constant in front only sets the overall scale, so the plots show the
# pattern normalized to its value straight ahead.


def lienard_pattern(
    n: np.ndarray, beta_vec: np.ndarray, accel_dir: np.ndarray
) -> np.ndarray:
    """
    Evaluate the Lienard angular pattern for any velocity and acceleration.

    Parameters
    ----------
    n : np.ndarray
        Unit observation vectors, shape (..., 3).
    beta_vec : np.ndarray
        Velocity divided by c, as a 3-vector.
    accel_dir : np.ndarray
        Unit vector along the acceleration.

    Returns
    -------
    np.ndarray
        |n x ((n - beta) x a_hat)|^2 / (1 - n . beta)^5 at each direction.
        Multiplying by q^2 a^2 / (16 pi^2 eps0 c^3) gives dP/dOmega.
    """
    n_dot_beta = n @ beta_vec
    inner = np.cross(n - beta_vec, accel_dir)
    outer = np.cross(n, inner)
    return np.sum(outer**2, axis=-1) / (1 - n_dot_beta) ** 5


def in_plane_pattern(theta: np.ndarray, beta: float) -> np.ndarray:
    """
    Evaluate the pattern in the orbit plane, at angle theta from the velocity.

    Parameters
    ----------
    theta : np.ndarray
        Angle from the velocity, in radians, measured toward the acceleration.
    beta : float
        Electron speed as a fraction of c.

    Returns
    -------
    np.ndarray
        The unnormalized pattern at each angle, from the closed form
        [1 - sin^2(theta) / (gamma^2 (1 - beta cos(theta))^2)] / (1 - beta cos(theta))^3.
        This is what lienard_pattern reduces to in the orbit plane, and it is
        much faster, which matters when the map is redrawn every frame.
    """
    gamma_sq = 1 / (1 - beta**2)
    doppler = 1 - beta * np.cos(theta)
    return (1 - np.sin(theta) ** 2 / (gamma_sq * doppler**2)) / doppler**3


def null_angle(beta: float) -> float:
    """
    Return the angle from the velocity (radians) at which the in-plane pattern vanishes.

    In the orbit plane the bracket 1 - sin^2(theta) / (gamma^2 (1 - beta cos(theta))^2)
    is zero where sin(theta) = gamma (1 - beta cos(theta)), which solves to
    sin(theta_null) = 1 / gamma. For a slow electron that is 90 degrees, the
    null of a dipole along its own axis. For a fast one it is 1 / gamma.
    """
    return float(np.arcsin(np.sqrt(1 - beta**2)))


def power_fractions(beta: float) -> tuple[float, float]:
    """
    Integrate the pattern over the whole sphere.

    Parameters
    ----------
    beta : float
        Electron speed as a fraction of c.

    Returns
    -------
    tuple[float, float]
        The integral of the pattern over all solid angle, and the fraction of
        it that lies inside the cone theta < 1 / gamma around the velocity.
    """
    gamma = 1 / np.sqrt(1 - beta**2)
    theta = np.linspace(0, np.pi, 4_001)
    phi = np.linspace(0, 2 * np.pi, 181)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    n = np.stack(
        [np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)], axis=-1
    )
    pattern = lienard_pattern(n, np.array([beta, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    per_theta = trapezoid(pattern, phi, axis=1) * np.sin(theta)
    total = float(trapezoid(per_theta, theta))
    inside = theta <= 1 / gamma
    in_cone = float(trapezoid(per_theta[inside], theta[inside]))
    return total, in_cone / total


def orbit_plane_intensity(
    x: np.ndarray,
    y: np.ndarray,
    position: np.ndarray,
    velocity_dir: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Radiation pattern in the orbit plane for an electron at a given point.

    Every grid point is colored by dP/dOmega along the line from the electron
    to that point. The 1 / r^2 falloff of a real far field is left out so
    that the headlight shows as a beam of uniform brightness, which is the
    picture that matters for a sweeping searchlight.

    Parameters
    ----------
    x, y : np.ndarray
        Grid coordinates of the plane.
    position : np.ndarray
        Electron position (x, y).
    velocity_dir : np.ndarray
        Unit vector (x, y) along the velocity. The acceleration is at right
        angles to it in the plane, which the in-plane formula already assumes.
    beta : float
        Electron speed as a fraction of c.

    Returns
    -------
    np.ndarray
        The pattern divided by its value straight ahead.
    """
    dx = x - position[0]
    dy = y - position[1]
    # Angle between the line of sight and the velocity. The in-plane pattern
    # is the same on both sides of the velocity, so the sign does not matter
    theta = np.arctan2(
        dx * velocity_dir[1] - dy * velocity_dir[0],
        dx * velocity_dir[0] + dy * velocity_dir[1],
    )
    # The pattern straight ahead is 1 / (1 - beta)^3, the brightest direction
    return in_plane_pattern(theta, beta) * (1 - beta) ** 3


def main():
    # TODO: Change the electron speed (fraction of c, 0 <= beta < 1)
    beta = 0.9

    # TODO: Set True to watch the electron circulate and the beam sweep the ring
    animate = False

    gamma = 1 / np.sqrt(1 - beta**2)
    theta_null = null_angle(beta)
    total, in_cone = power_fractions(beta)

    print(f"Electron speed:                 beta  = {beta:.4f}")
    print(f"Lorentz factor:                 gamma = {gamma:.4f}")
    print(
        f"Nulls in the orbit plane:       theta_null = arcsin(1/gamma) = "
        f"{np.degrees(theta_null):.3f} deg   (1/gamma = {np.degrees(1 / gamma):.3f} deg)"
    )
    print(
        f"Forward / backward brightness:  ((1+beta)/(1-beta))^3 = {((1 + beta) / (1 - beta)) ** 3:,.1f}"
    )
    print(
        f"Integral over the sphere divided by (8 pi / 3) gamma^4 = "
        f"{total / (8 * np.pi / 3 * gamma**4):.5f}   (expected 1)"
    )
    print(f"Share of the power inside the cone theta < 1/gamma: {in_cone:.3f}")

    # The orbit-plane map uses the closed form; confirm it matches the vector formula
    check_theta = np.linspace(0, np.pi, 3_601)
    check_n = np.stack(
        [np.cos(check_theta), np.sin(check_theta), np.zeros_like(check_theta)], axis=-1
    )
    vector_form = lienard_pattern(
        check_n, np.array([beta, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    closed_form = in_plane_pattern(check_theta, beta)
    print(
        "Largest difference between the closed form and the vector formula in the "
        f"orbit plane: {np.max(np.abs(closed_form - vector_form)) * (1 - beta) ** 3:.1e}"
        " (relative to straight ahead)"
    )

    # The ring, drawn in the orbit plane with the electron starting at the bottom
    ring_radius = 0.5
    extent = 1.2
    n_points = 601
    x, y = np.meshgrid(
        np.linspace(-extent, extent, n_points),
        np.linspace(-extent, extent, n_points),
    )
    norm = LogNorm(vmin=1e-4, vmax=1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, num=Path(__file__).name, figsize=(15, 7), layout="constrained"
    )
    fig.suptitle(
        "Synchrotron radiation: an electron on a circular orbit\n"
        rf"$\beta$ = {beta:.3f},  $\gamma$ = {gamma:.3f},  nulls at "
        rf"$\theta$ = $\pm$arcsin(1/$\gamma$) = $\pm${np.degrees(theta_null):.1f}$^\circ$"
    )

    # Left panel: the orbit plane. Every artist that moves with the electron is
    # created once here and repositioned by update()
    image = ax1.imshow(
        np.zeros_like(x),
        cmap="inferno",
        norm=norm,
        origin="lower",
        extent=(-extent, extent, -extent, extent),
    )
    ax1.add_patch(
        Circle((0, 0), ring_radius, fill=False, ls="--", color="lightgray", lw=1)
    )
    ax1.text(0, 0, "ring", color="lightgray", ha="center", va="center")
    (electron_dot,) = ax1.plot([], [], "o", color="deepskyblue", ms=8, zorder=5)
    (lobe_line,) = ax1.plot(
        [], [], color="cyan", lw=1.2, label=r"d$P$/d$\Omega$ (polar plot)"
    )
    null_lines = [
        ax1.plot(
            [], [], color="white", ls="--", lw=1, label="nulls" if i == 0 else None
        )[0]
        for i in range(2)
    ]
    arrows = {}
    labels = {}
    for key, text in (("v", r"$\vec{v}$"), ("a", r"$\vec{a}$")):
        arrows[key] = ax1.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "deepskyblue", "lw": 2},
            zorder=6,
        )
        labels[key] = ax1.text(0, 0, text, color="deepskyblue", fontsize=12, zorder=6)

    # The polar overlay keeps its shape; only its orientation and center change
    phi = np.linspace(-np.pi, np.pi, 2_000)
    lobe_shape = 0.45 * in_plane_pattern(phi, beta) * (1 - beta) ** 3

    def update(frame: int) -> list:
        """Move the electron to its position for this frame and redraw everything that follows it."""
        ring_angle = np.radians(-90 + 2 * frame)
        radial = np.array([np.cos(ring_angle), np.sin(ring_angle)])
        position = ring_radius * radial
        velocity_dir = np.array([-radial[1], radial[0]])  # counterclockwise
        accel_dir = -radial  # toward the center
        heading = np.arctan2(velocity_dir[1], velocity_dir[0])

        image.set_data(orbit_plane_intensity(x, y, position, velocity_dir, beta))
        electron_dot.set_data([position[0]], [position[1]])
        lobe_line.set_data(
            position[0] + lobe_shape * np.cos(heading + phi),
            position[1] + lobe_shape * np.sin(heading + phi),
        )
        for line, sign in zip(null_lines, (1, -1)):
            angle = heading + sign * theta_null
            line.set_data(
                [position[0], position[0] + 2 * extent * np.cos(angle)],
                [position[1], position[1] + 2 * extent * np.sin(angle)],
            )
        for key, direction, length in (("v", velocity_dir, 0.3), ("a", accel_dir, 0.2)):
            arrows[key].xy = position + length * direction
            arrows[key].set_position(position)
            labels[key].set_position(position + (length + 0.03) * direction)
        return [
            image,
            electron_dot,
            lobe_line,
            *null_lines,
            *arrows.values(),
            *labels.values(),
        ]

    update(0)
    ax1.set_title("The orbit plane: a headlight tangent to the ring")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(-extent, extent)
    ax1.set_ylim(-extent, extent)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower left", facecolor="black", labelcolor="white")
    fig.colorbar(
        image,
        ax=ax1,
        shrink=0.8,
        label=r"d$P$/d$\Omega$ along the line from the electron, relative to straight ahead",
    )

    # Right panel: what a screen facing the oncoming electron sees. Angles are
    # measured from the velocity, horizontal in the orbit plane and vertical
    # out of it, in units of 1 / gamma
    half_width = min(4.0, 1.5 * gamma)
    angles = np.linspace(-half_width, half_width, 601) / gamma
    h, v = np.meshgrid(angles, angles)
    n_screen = np.stack(
        [np.cos(h) * np.cos(v), np.sin(h) * np.cos(v), np.sin(v)], axis=-1
    )
    screen = (
        lienard_pattern(n_screen, np.array([beta, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        * (1 - beta) ** 3
    )
    screen_norm = LogNorm(vmin=1e-4, vmax=1)
    screen_image = ax2.imshow(
        np.clip(screen, screen_norm.vmin, None),
        cmap="inferno",
        norm=screen_norm,
        origin="lower",
        extent=(-half_width, half_width, -half_width, half_width),
    )
    ax2.axhline(0, color="white", ls=":", lw=0.8)
    ax2.text(-half_width * 0.97, 0.08, "orbit plane", color="white", fontsize=9)
    ax2.add_patch(
        Circle(
            (0, 0),
            1,
            fill=False,
            ls="--",
            color="white",
            lw=1,
            label=r"cone $\theta$ = 1/$\gamma$",
        )
    )
    ax2.plot(
        [-gamma * theta_null, gamma * theta_null],
        [0, 0],
        "x",
        color="cyan",
        ms=10,
        mew=2,
        label="nulls",
    )
    ax2.set_title(
        "A screen facing the electron: two nulls pinch the beam in the orbit plane"
    )
    ax2.set_xlabel(
        r"horizontal angle from $\vec{v}$, in units of 1/$\gamma$  (in the orbit plane)"
    )
    ax2.set_ylabel(r"vertical angle from $\vec{v}$, in units of 1/$\gamma$")
    ax2.set_aspect("equal")
    ax2.legend(loc="lower left", facecolor="black", labelcolor="white")
    fig.colorbar(
        screen_image,
        ax=ax2,
        shrink=0.8,
        label=r"d$P$/d$\Omega$ relative to straight ahead",
    )

    if animate:
        # Keep a reference so the animation is not garbage collected
        animation = FuncAnimation(fig, update, frames=180, interval=40, blit=False)
        fig.animation = animation
    plt.show()


if __name__ == "__main__":
    main()
