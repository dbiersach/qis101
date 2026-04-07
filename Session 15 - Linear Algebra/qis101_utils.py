"""
Utility functions for the QIS101 course.

Provides LaTeX rendering helpers for NumPy arrays in Jupyter notebooks.
"""

import numpy as np
from IPython.display import Math

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
