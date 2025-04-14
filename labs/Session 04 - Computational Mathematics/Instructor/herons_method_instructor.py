#!/usr/bin/env python3
"""herons_method_instructor.py"""


def sqrt(s):
    x = s / 2
    while x**2 - s > 1e-10:
        x = (s / x + x) / 2
    return x


def main():
    x = 168923.74
    print(f"Square root of {x} = {sqrt(x)}")


if __name__ == "__main__":
    main()
