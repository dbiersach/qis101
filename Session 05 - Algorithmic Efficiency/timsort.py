#!/usr/bin/env -S uv run
"""timsort.py"""

import random
import time


def main():
    random.seed(2016)
    num_samples = 10_000
    print(f"Timsort {num_samples:,} random samples...")

    samples = [random.randint(1, 10_000) for _ in range(num_samples)]

    print("UNSORTED")
    print(f"{samples[:10]} ... {samples[-10:]}")

    start_time = time.perf_counter()
    samples.sort()
    elapsed_time = time.perf_counter() - start_time

    print("SORTED")
    print(f"{samples[:10]} ... {samples[-10:]}")

    print(f"Total run time (sec): {elapsed_time:.3f}\n")


if __name__ == "__main__":
    main()
