#!/usr/bin/env -S uv run
"""quicksort.py"""

import random
import time


def median_of_three(values, low, high):
    # Finds the index of the median value
    # between the low, high, and middle elements
    mid = (low + high) // 2
    x = values[low]
    y = values[mid]
    z = values[high]
    if x < y:
        if y < z:
            return mid
        elif x < z:
            return high
        else:
            return low
    else:
        if x < z:
            return low
        elif y < z:
            return high
        else:
            return mid


def quick_sort_partition(values, low_index, high_index):
    # Partitions a list using the median of three strategy
    pivot_index = median_of_three(values, low_index, high_index)  # pivot index
    pivot_value = values[pivot_index]  # pivot value

    while True:
        while values[low_index] <= pivot_value and low_index < pivot_index:
            low_index += 1
        while values[high_index] > pivot_value and high_index > pivot_index:
            high_index -= 1
        if low_index == pivot_index and high_index == pivot_index:
            return pivot_index

        # Swap array values at lo and hi indexes
        values[low_index], values[high_index] = values[high_index], values[low_index]

        if low_index == pivot_index:
            pivot_index = high_index
        elif high_index == pivot_index:
            pivot_index = low_index


def quick_sort(values, low_index, high_index):
    # Sorts the provided list in increasing order using the Quicksort algorithm
    if low_index < high_index:
        partition_index = quick_sort_partition(values, low_index, high_index)
        if partition_index > 0:
            # Sort the left-hand sub-array
            quick_sort(values, low_index, partition_index - 1)
        # Sort the right-hand sub-array
        quick_sort(values, partition_index + 1, high_index)
    return values


def main():
    random.seed(2016)

    num_samples = 10_000
    print(f"Quicksort {num_samples:,} random samples...")

    samples = [random.randint(1, 10_000) for _ in range(num_samples)]

    print("UNSORTED")
    print(f"{samples[:10]} ... {samples[-10:]}")

    start_time = time.perf_counter()
    samples = quick_sort(samples, 0, len(samples) - 1)
    elapsed_time = time.perf_counter() - start_time

    print("SORTED")
    print(f"{samples[:10]} ... {samples[-10:]}")

    print(f"Total run time (sec): {elapsed_time:.3f}\n")


if __name__ == "__main__":
    main()
