#!/usr/bin/env -S uv run
"""euclid_gcd.py"""


def gcd(a, b):
    if a < b:
        a, b = b, a
    c = a - b
    while c > 0:
        if c > b:
            a = c
        else:
            a = b
            b = c
        c = a - b
    return b


def gcd_fast(a, b):
    while b > 0:
        a, b = b, a % b
    return a


def main():
    a, b = 182, 231
    print(f"The GCD of {a} and {b} = {gcd(a, b)}")
    print(f"The GCD of {a} and {b} = {gcd_fast(a, b)}")


if __name__ == "__main__":
    main()
