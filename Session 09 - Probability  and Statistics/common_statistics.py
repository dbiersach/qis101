#!/usr/bin/env python3
"""common_statistics.py"""

import collections
import random


def mean(s):
    return sum(s) / len(s)


def median(s):
    s.sort()
    i = len(s) // 2
    if len(s) % 2 == 1:
        return s[i]
    else:
        return (s[i - 1] + s[i]) / 2


def mode(s):
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
