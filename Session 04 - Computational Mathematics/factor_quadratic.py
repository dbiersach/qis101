#!/usr/bin/env -S uv run
"""factor_quadratic.py"""


def factor_quadratic(J: int, K: int, L: int) -> None:
    print("Given the quadratic:", end=" ")
    print(f"{J}x^2 + {K}x + {L}")


def main() -> None:
    factor_quadratic(115425, 3254121, 379020)


if __name__ == "__main__":
    main()
