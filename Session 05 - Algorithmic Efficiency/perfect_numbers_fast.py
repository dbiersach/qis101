#!/usr/bin/env -S uv run
"""perfect_numbers_fast.py"""

import time

import numpy as np
from tqdm import tqdm


def is_perfect(n):
    x = np.arange(1, n)
    factors = x[np.where(n % x == 0)]
    return np.sum(factors) == n


def perfect_numbers_slow(limit):
    found = []
    for n in tqdm(range(2, limit), desc="Slow method"):
        if is_perfect(n):
            found.append(n)
    return found


def perfect_numbers_fast(limit):
    found = []
    for n in tqdm(range(2, limit), desc="Fast method"):
        divisors = np.arange(2, int(np.sqrt(n)) + 1)
        divisors = divisors[n % divisors == 0]
        pairs = n // divisors
        all_factors = np.unique(np.concatenate([[1], divisors, pairs]))
        if np.sum(all_factors) == n:
            found.append(n)
    return found


def main():
    limit = 100_000
    print(f"Finding Perfect numbers up to {limit:,}:")

    start_time = time.perf_counter()
    found = perfect_numbers_slow(limit)
    elapsed_time_slow = time.perf_counter() - start_time
    print(f"{found}", sep=", ")
    print(f"Slow method run time (sec): {elapsed_time_slow:.3f}")

    start_time = time.perf_counter()
    found = perfect_numbers_fast(limit)
    elapsed_time_fast = time.perf_counter() - start_time
    print(f"{found}", sep=", ")
    print(f"Fast method run time (sec): {elapsed_time_fast:.3f}")


if __name__ == "__main__":
    main()
