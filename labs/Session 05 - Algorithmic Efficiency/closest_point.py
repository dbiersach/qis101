#!/usr/bin/env python3
"""closest_point.py"""

import random
from pprint import pprint


def generate_points(n, x_range, y_range):
    """Generate n random points within the given range."""
    random.seed(2022)
    return [
        (round(random.uniform(*x_range), 4), round(random.uniform(*y_range), 4))
        for _ in range(n)
    ]


# Generate random points
num_points = 10_000
points = generate_points(num_points, (0, 100), (0, 100))
pprint(points[:5])
print(f"Finding closest point among {num_points:,} random points...")
print()
