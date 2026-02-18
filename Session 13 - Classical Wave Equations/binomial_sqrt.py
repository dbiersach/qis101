#!/usr/bin/env -S uv run
"""binomial_sqrt.py"""

from math import pow
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import Float, Integer, Number, Poly, lambdify, symbols


def expr_to_str(expr: Poly, num_digits: int) -> str:
    """
    Returns a string representation of the given Polynomial expression (expr),
    rounding each coefficient to a fractional part having (num_digits) precision
    """
    symbolic_expr = expr.as_expr()  # Converts Poly to a plain symbolic expression
    replacements = {  # Dictionary comprehension
        n: n if isinstance(n, Integer) else Float(n.evalf(), num_digits)
        for n in symbolic_expr.atoms(Number)
    }
    rounded_expr = symbolic_expr.xreplace(replacements)
    return str(rounded_expr)


def calc_coeff(a: float, b: float, r: float, n: int) -> float:
    """
    Returns the coefficient for the (n)th term in the Binomial expansion of (a+b)^r
    """
    coeff = 1.0
    for m in range(n):
        coeff = coeff * (r - m) / (m + 1)
    coeff = coeff * pow(a, r - n)
    coeff = coeff * pow(b, n)
    return coeff


def binomial_expand(
    a: float, b: float, c: float, r: float, max_t: int
) -> tuple[Poly, Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    """
    Returns a tuple containing the Binomial Expansion of (a+b*x^c)^r
    to (max_t) terms as a Sympy Polynomial in x, along with
    a callable Numpy expression for that expansion
    """
    x = symbols("x")
    poly = 0.0
    for t in range(max_t):
        # Append this term (as a symbolic expression in x)
        # to the growing polynomial of max_t terms
        poly += calc_coeff(a, b, r, t) * x ** (c * t)
    return poly, lambdify(x, poly.as_expr(), modules="numpy")


def heron(s: float) -> tuple[int, float]:
    """
    Returns a tuple containing the number of iterations (n) of Heron's Method
    to calculate sqrt(s) to within 1e-14, along with the actual root value
    """
    x, t = s / 2, 1
    while x**2 - s > 1e-14:
        x = (s / x + x) / 2
        t += 1
    return t, x


def main():
    print(f"{'Terms':>5}{'Estimate':>18}{'Binomial Expansion':>21}")
    for terms in range(1, 21):
        eqn = binomial_expand(1, -1, 1, 1 / 2, terms)
        # Evaluate the symbolic expression at x = 2/9
        print(f"{terms:>5}  {3 * eqn[1](2 / 9):.14f}", end="")
        if terms < 8:
            print(f" = 3*({expr_to_str(eqn[0], 5)})", end=" ")
        print()

    # Compare the Binomial Expression convergence rate to Heron's Method
    terms, x = heron(7)
    print("Heron's Method")
    print(f"{terms:>5}  {x:.14f}", end="")


if __name__ == "__main__":
    main()
