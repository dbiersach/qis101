#!/usr/bin/env python3
"""frobenius_serial.py"""

a1, a2, a3 = 6, 9, 20
limit = a1 * a2 * a3
can_make = set()

# Try all combinations
for i in range(limit // a1 + 1):
    for j in range(limit // a2 + 1):
        for k in range(limit // a3 + 1):
            value = a1 * i + a2 * j + a3 * k
            can_make.add(value)

# Find values up to the limit that CANNOT be made
cannot_make = set(range(limit)) - can_make
print(cannot_make)
