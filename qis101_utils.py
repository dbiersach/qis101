"""qis101_utils.py

Shared utility functions for the QIS101 course.

Sections
--------
1. LaTeX rendering helpers for NumPy arrays in Jupyter notebooks.
2. Ideal-pendulum physics and symplectic integrators used across
    Session 19 dynamical-systems scripts.
"""

import numpy as np
from IPython.display import Math

# ── Section 1: LaTeX rendering ────────────────────────────────────────────────

# (value, LaTeX string) pairs for recognizing common exact fractions.
# Comparisons are made against abs(np.round(val, 5)).
_SPECIAL_VALUES: list[tuple[float, str]] = [
    (0.25000, r"\frac{1}{4}"),
    (0.50000, r"\frac{1}{2}"),
    (0.57735, r"\frac{1}{\sqrt{3}}"),
    (0.70711, r"\frac{1}{\sqrt{2}}"),
    (0.81650, r"\sqrt{\frac{2}{3}}"),
    (0.86603, r"\frac{\sqrt{3}}{2}"),
]


def _format_real(val: float, places: int = 5) -> str:
    """Return a LaTeX string for a real scalar, recognizing common fractions."""
    rounded = abs(np.round(val, 5))
    for threshold, latex in _SPECIAL_VALUES:
        if rounded == threshold:
            return ("-" if val < 0 else "") + latex

    fmt = f"{{v:.{places}f}}"
    result = fmt.format(v=val).rstrip("0").rstrip(".")
    return "0" if float(result) == 0 else result


def as_latex(
    a: np.ndarray,
    places: int = 5,
    column: bool = False,
    prefix: str = "",
) -> Math:
    """Render a NumPy array as a LaTeX bmatrix for display in a Jupyter notebook.

    Args:
        a:       1-D or 2-D array of real or complex values.
        places:  Decimal places used when formatting non-special values.
        column:  If True, treat a 1-D array as a column vector.
        prefix:  Optional LaTeX string prepended before the bmatrix.

    Returns:
        An IPython ``Math`` object ready for ``display()``.
    """
    matrix = np.copy(a)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
        if column:
            matrix = matrix.T

    precision = 1 / 10**places
    rows: list[str] = []

    for row in range(matrix.shape[0]):
        cells: list[str] = []
        for col in range(matrix.shape[1]):
            cell = matrix[row, col]
            real_part = float(np.real(cell))
            imag_part = float(np.imag(cell))

            is_imag_neg = imag_part < 0
            is_real_zero = np.isclose(real_part, 0, atol=precision)
            is_imag_zero = np.isclose(imag_part, 0, atol=precision)
            is_imag_one = np.isclose(abs(imag_part), 1, atol=precision)

            cell_parts: list[str] = []
            if is_real_zero and is_imag_zero:
                cell_parts.append("0")
            elif not is_real_zero:
                cell_parts.append(_format_real(real_part, places))

            if not is_imag_zero:
                if is_imag_one:
                    cell_parts.append(
                        "-i" if is_imag_neg else ("+" if not is_real_zero else "") + "i"
                    )
                else:
                    if not is_real_zero and not is_imag_neg:
                        cell_parts.append(" + ")
                    cell_parts.append(_format_real(imag_part, places) + "i")

            cells.append("".join(cell_parts))

        rows.append(" & ".join(cells))

    latex_body = r" \\[1em] ".join(rows)
    return Math(prefix + r"\begin{bmatrix}" + latex_body + r"\\" + r"\end{bmatrix}")


# ── Section 2: Pendulum physics and symplectic integrators ────────────────────

# Physical constants for an ideal simple pendulum
PENDULUM_LENGTH = 1.0  # pendulum length (m)
PENDULUM_G = 9.81  # gravitational acceleration (m/s²)


def pendulum_angular_acceleration(theta: float | np.ndarray) -> float | np.ndarray:
    """Compute the angular acceleration of an ideal pendulum.

    Applies the exact (nonlinear) equation of motion: α = -(g/L) sin(θ),
    with no small-angle approximation.

    Parameters
    ----------
    theta : float or ndarray
        Angular displacement from vertical (radians)

    Returns
    -------
    float or ndarray
        Angular acceleration (rad/s²)
    """
    return -PENDULUM_G / PENDULUM_LENGTH * np.sin(theta)


def pendulum_total_energy(
    theta: float | np.ndarray, omega: float | np.ndarray
) -> float | np.ndarray:
    """Compute the total mechanical energy of the pendulum per unit mass.

    Returns the sum of kinetic and potential energy in the form:
    E = (1/2)ω² - (g/L)cos(θ), where potential energy is referenced
    to the pivot point.

    Parameters
    ----------
    theta : float or ndarray
        Angular displacement from vertical (radians)
    omega : float or ndarray
        Angular velocity (rad/s)

    Returns
    -------
    float or ndarray
        Total mechanical energy per unit mass (J/kg)
    """
    return 0.5 * omega**2 - PENDULUM_G / PENDULUM_LENGTH * np.cos(theta)


def pendulum_euler_cromer(
    theta0: float, omega0: float, t_final: float, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the pendulum equations using the Euler-Cromer method.

    A first-order symplectic integrator that updates velocity before position,
    giving exact energy conservation on average. Superior to standard Euler
    for oscillatory systems despite being only first-order accurate.

    Parameters
    ----------
    theta0 : float
        Initial angular displacement (radians)
    omega0 : float
        Initial angular velocity (rad/s)
    t_final : float
        Total integration time (s)
    dt : float
        Fixed time step size (s)

    Returns
    -------
    t : ndarray
        Time array (s)
    theta : ndarray
        Angular displacement at each time step (radians)
    omega : ndarray
        Angular velocity at each time step (rad/s)
    """
    n_steps = int(t_final / dt)
    t = np.arange(n_steps) * dt
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0], omega[0] = theta0, omega0
    for i in range(n_steps - 1):
        alpha = pendulum_angular_acceleration(theta[i])
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt
    return t, theta, omega


def pendulum_velocity_verlet(
    theta0: float, omega0: float, t_final: float, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the pendulum equations using the Velocity Verlet method.

    A second-order symplectic integrator that evaluates acceleration at both
    the current and next positions, achieving better energy conservation and
    phase accuracy than Euler-Cromer with the same step size.

    Parameters
    ----------
    theta0 : float
        Initial angular displacement (radians)
    omega0 : float
        Initial angular velocity (rad/s)
    t_final : float
        Total integration time (s)
    dt : float
        Fixed time step size (s)

    Returns
    -------
    t : ndarray
        Time array (s)
    theta : ndarray
        Angular displacement at each time step (radians)
    omega : ndarray
        Angular velocity at each time step (rad/s)
    """
    n_steps = int(t_final / dt)
    t = np.arange(n_steps) * dt
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0], omega[0] = theta0, omega0
    for i in range(n_steps - 1):
        alpha = pendulum_angular_acceleration(theta[i])
        theta[i + 1] = theta[i] + omega[i] * dt + 0.5 * alpha * dt**2
        alpha_new = pendulum_angular_acceleration(theta[i + 1])
        omega[i + 1] = omega[i] + 0.5 * (alpha + alpha_new) * dt
    return t, theta, omega


def yoshida_coeffs() -> tuple[np.ndarray, np.ndarray]:
    """Compute the Yoshida 4th-order symplectic integrator coefficients.

    Derives the position (c) and velocity (d) substep coefficients from
    Yoshida (1990), "Construction of higher order symplectic integrators,"
    Physics Letters A, 150(5-7), 262-268.

    Returns
    -------
    c : ndarray, shape (4,)
        Position half-step coefficients
    d : ndarray, shape (3,)
        Velocity full-step coefficients
    """
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)
    c = np.array([w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0])
    d = np.array([w1, w0, w1])
    return c, d


def pendulum_yoshida4(
    theta0: float, omega0: float, t_final: float, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the pendulum equations using the Yoshida 4th-order symplectic method.

    A fourth-order symplectic integrator constructed from three Verlet substeps
    with carefully chosen coefficients (Forest & Ruth 1990 / Yoshida 1990).
    Provides superior long-term energy conservation and spectral purity compared
    to lower-order symplectic methods.

    Parameters
    ----------
    theta0 : float
        Initial angular displacement (radians)
    omega0 : float
        Initial angular velocity (rad/s)
    t_final : float
        Total integration time (s)
    dt : float
        Fixed time step size (s)

    Returns
    -------
    t : ndarray
        Time array (s)
    theta : ndarray
        Angular displacement at each time step (radians)
    omega : ndarray
        Angular velocity at each time step (rad/s)
    """
    c, d = yoshida_coeffs()

    n_steps = int(t_final / dt)
    t = np.arange(n_steps) * dt
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0], omega[0] = theta0, omega0

    for i in range(n_steps - 1):
        th, om = theta[i], omega[i]
        for j in range(3):
            th += c[j] * om * dt
            om += d[j] * pendulum_angular_acceleration(th) * dt
        th += c[3] * om * dt
        theta[i + 1], omega[i + 1] = th, om

    return t, theta, omega


def main():
    print("This module is intended to be imported, not executed directly")


if __name__ == "__main__":
    main()
