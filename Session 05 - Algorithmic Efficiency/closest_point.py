#!/usr/bin/env -S uv run
"""closest_point.py"""

import random
import time
from pprint import pprint


def generate_points(n, x_range, y_range):
    """Generate n random points within the given range."""
    random.seed(2022)
    return [
        (round(random.uniform(*x_range), 4), round(random.uniform(*y_range), 4))
        for _ in range(n)
    ]


def closest_pair(points):
    min_dist = float("inf")
    closest_pair = ((0, 0), (0, 0))
    return closest_pair, min_dist


def main():
    # Generate random points
    num_points = 10_000
    print(f"Generating {num_points:,} random points...")
    points = generate_points(num_points, (0, 100), (0, 100))
    print("The first five random points are:")
    pprint(points[:5])
    print("Finding the closest pair of points:")

    # Measure time to find closest pair of points
    start_time = time.perf_counter()
    result = closest_pair(points)
    elapsed_time = time.perf_counter() - start_time
    print("Nearest Points", end=": ")
    print(f"{result[0][0]} - {result[0][1]}")
    print(f"Minimum distance = {result[1]:.4f}")
    print(f"Time taken: {elapsed_time:.4f}")


if __name__ == "__main__":
    main()
