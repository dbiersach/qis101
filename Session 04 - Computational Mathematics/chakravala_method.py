#!/usr/bin/env -S uv run
"""chakravala_method.py"""

from math import isqrt


def chakravala_solution(n):
    # Check if n is a perfect square
    r = isqrt(n)
    if r * r == n:
        # No solution for perfect squares
        return 0, 0

    # Initialize first triple (a, b, k)
    a = r + 1
    b = 1
    # From auxiliary equation: a^2 - n * b^2 = k
    k = a * a - n * b * b

    # Apply the Chakravala method until k becomes 1
    while k != 1:
        # Among nonnegative m with (a + b*m) divisible by k, minimize
        # abs(m*m - n). Only the nearest eligible m on each side of
        # sqrt(n) can win; each side repeats every abs(k) integers.
        m0 = isqrt(n)
        modulus = abs(k)
        low = max(0, m0 - modulus + 1)
        high = m0 + modulus + 1
        candidates = (m for m in range(low, high) if (a + b * m) % k == 0)
        best_m = min(candidates, key=lambda m: abs(m * m - n))

        # Update (a, b, k) using the best m found
        m, a_old, b_old, k_old = best_m, a, b, k
        a = (a_old * m + n * b_old) // abs(k_old)
        b = (a_old + b_old * m) // abs(k_old)
        k = (m * m - n) // k_old

    return a, b


def main():
    print(f"{'n':>4}{'x':>12}{'y':>12}")
    print(f"{'=' * 3:>4}{'=' * 10:>12}{'=' * 10:>12}")

    for i in range(2, 71):
        x, y = chakravala_solution(i)
        if x > 0 and y > 0:
            print(f"{i:>4}{x:>12}{y:>12}")
        else:
            print(f"{i:>4}{'-':>12}{'-':>12}")


if __name__ == "__main__":
    main()
