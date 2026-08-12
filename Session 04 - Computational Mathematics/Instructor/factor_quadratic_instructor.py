#!/usr/bin/env -S uv run
"""factor_quadratic_instructor.py"""


def factor_quadratic(J: int, K: int, L: int) -> None:
    print("Given the quadratic:", end=" ")
    print(f"{J}x^2 + {K}x + {L}")

    for a in range(1, J + 1):
        if J % a == 0:
            c: int = J // a
            for b in range(1, L + 1):
                if L % b == 0:
                    d: int = L // b
                    if a * d + b * c == K:
                        print("The factors are:", end=" ")
                        print(f"({a}x + {b})({c}x + {d})")


def main() -> None:
    factor_quadratic(115425, 3254121, 379020)


if __name__ == "__main__":
    main()
