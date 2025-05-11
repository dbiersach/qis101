#!/usr/bin/env python3
"""newton_binomial.py"""

from math import pow
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
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


def main():
    x = np.linspace(0, 10, 1000)

    plt.figure(Path(__file__).name)
    plt.plot(x, 1 / np.power(2 * np.power(x, 2) + 7, 2 / 3), label="Exact")

    print(f"{'Terms':>5}   Binomial Expansion")
    for t in range(2, 8):
        # Use Newton's Binomial Theorem to expand (2x^2+7)^(-2/3) to 't' terms
        eqn = binomial_expand(7, 2, 2, -2 / 3, t)
        print(f"{t:>5} = {expr_to_str(eqn[0], 5)}")
        # Evaluate the symbolic expression across the domain x=[0, 10]
        plt.plot(x, np.array(list(map(eqn[1], x))), label=f"{t} terms")

    plt.title(r"Binomial Expansion of $y=(2x^2+7)^{-2/3}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, 0.3)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
