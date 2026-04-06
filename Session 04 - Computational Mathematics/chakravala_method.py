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
        """
        Find m such that both conditions are true:
            (a + b * m) % k == 0 -> to ensure the next b is an integer
            m^2 - n is minimized -> to reduce the next k as much as possible
        """
        m0 = isqrt(n)
        best_m = 0
        min_delta = 0
        found_m = False

        # Search for m around m0
        for t in range(abs(k) + 1):
            # Check both sides of m0
            for m in (m0 - t, m0 + t):
                # Verify divisibility requirement
                if (a + b * m) % k == 0:
                    # We have a potential m, calculate the gap
                    delta = m**2 - n
                    # Update best m if this is the first m or if delta is smaller
                    if not found_m or delta < min_delta:
                        best_m = m
                        min_delta = delta
                        found_m = True
            # Stop searching once we find a valid m
            if found_m:
                break

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
