#!/usr/bin/env -S uv run
"""herons_formula.py"""

from math import sqrt
from random import randint


def is_triangle(triangle: tuple[int, int, int]) -> bool:
    """Determine if a triangle is non-degenerate"""
    a, b, c = triangle
    return a + b > c and a + c > b and b + c > a


def area(triangle: tuple[int, int, int]) -> float:
    """Returns area of a triangle given a tuple of its three side lengths"""
    a, b, c = triangle
    s: float = (a + b + c) / 2
    return sqrt(s * (s - a) * (s - b) * (s - c))


def main() -> None:
    for _ in range(10):
        while not is_triangle(t := (randint(1, 100), randint(1, 100), randint(1, 100))):
            continue
        print(f"{t} {area(t):>9.4f}")


if __name__ == "__main__":
    main()
