#!/usr/bin/env -S uv run
"""uniform_variance.py"""

import numpy as np


def generate_set(set_num):
    size = np.random.randint(10_000, 200_000)
    lower_limit = np.random.randint(10_000)
    upper_limit = lower_limit + np.random.randint(100_000)
    samples = np.random.uniform(lower_limit, upper_limit, size)
    m, v = np.mean(samples), np.var(samples)
    magic_number = (upper_limit - lower_limit) ** 2 / v
    print(
        f"{set_num:>8}{size:>9,}{lower_limit:>9,}{upper_limit:>9,}"
        f"{m:>12.3f}{v:>16.3f}{magic_number:>9.3f}"
    )


def main():
    print(
        f"{'Set #':>8}{'Size':>9}{'Lower':>9}{'Upper':>9}"
        f"{'Mean':>12}{'Variance':>16}{'Magic':>9}"
    )

    for set_num in range(1, 16):
        generate_set(set_num)


if __name__ == "__main__":
    main()
