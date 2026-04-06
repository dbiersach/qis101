#!/usr/bin/env -S uv run
"""common_statistics.py"""

import collections
import random

# These functions are for demonstration purposes only.
# In practice, you should use the built-in functions in the statistics module.


def mean(s):
    # See statistics.mean()
    return sum(s) / len(s)


def median(s):
    # See statistics.median()
    t = sorted(s)
    i = len(t) // 2
    if len(t) % 2 == 1:
        return t[i]
    else:
        return (t[i - 1] + t[i]) / 2


def mode(s):
    # See statistics.mode() and statistics.multimode()
    c = collections.Counter(s)
    max_c = max(c.values())
    if max_c == 1:
        return []
    else:
        return [k for k, v in c.items() if v == max_c]


def main():
    a = [random.randint(1, 100) for _ in range(30)]

    print("a = ", end="{")
    print(*a, sep=", ", end="}\n")

    print(f"Mean of a = {mean(a):.2f}")
    print(f"Median of a = {median(a):.2f}")

    print("Mode of a = ", end="{")
    print(*mode(a), sep=", ", end="}\n")


if __name__ == "__main__":
    main()
