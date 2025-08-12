#!/usr/bin/env python3
"""prime_racer1.py"""

import random
import time


def is_prime(n):
    return all(n % factor != 0 for factor in range(2, n))


def main():
    random.seed(2016)
    num_primes = 0

    start_time = time.perf_counter()
    for _ in range(100_000):
        n = random.randint(1_000, 10_000)
        if is_prime(n):
            num_primes += 1
    elapsed_time = time.perf_counter() - start_time

    print(f"Number of primes found: {num_primes:,}")
    print(f"Total run time (sec): {elapsed_time:.3f}")


if __name__ == "__main__":
    main()
