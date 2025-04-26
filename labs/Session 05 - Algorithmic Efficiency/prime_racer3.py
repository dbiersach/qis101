#!/usr/bin/env python3
"""prime_racer3.py"""

import math
import random
import time


def is_prime(n):
    if n % 2 == 0:
        return False
    else:
        return all(n % factor != 0 for factor in range(3, int(math.sqrt(n)) + 1, 2))


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
