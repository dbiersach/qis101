#!/usr/bin/env -S uv run
"""uniform_variance.py"""

import numpy as np


def generate_set(set_num):
    size = np.random.randint(10_000, 200_000)
    lower_limit = np.random.randint(10_000)
    upper_limit = lower_limit + np.random.randint(100_000)
    set = np.random.uniform(lower_limit, upper_limit, size)
    m, v = np.mean(set), np.var(set)
    magic_number = (upper_limit - lower_limit) ** 2 / v
    print(f"{set_num:>8}", end="")
    print(f"{size:>9,}", end="")
    print(f"{lower_limit:>9,}", end="")
    print(f"{upper_limit:>9,}", end="")
    print(f"{m:>12.3f}", end="")
    print(f"{v:>16.3f}", end="")
    print(f"{magic_number:>9.3f}")


def main():
    print(f"{'Set #':>8}", end="")
    print(f"{'Size':>9}", end="")
    print(f"{'Lower':>9}", end="")
    print(f"{'Upper':>9}", end="")
    print(f"{'Mean':>12}", end="")
    print(f"{'Variance':>16}", end="")
    print(f"{'Magic':>9}")

    for set_num in range(1, 16):
        generate_set(set_num)


if __name__ == "__main__":
    main()
