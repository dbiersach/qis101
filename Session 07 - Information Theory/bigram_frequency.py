#!/usr/bin/env -S uv run
"""bigram_frequency.py"""

from collections import Counter
from pathlib import Path

FILE_NAME = "bigram_ciphertext.txt"
print(f'Bigram analysis of file \n"{FILE_NAME}" :')

file_path = Path(__file__).parent / FILE_NAME
with open(file_path, "rb") as f_in:
    f_bytes = bytearray(f_in.read())

# Create Counter dictionary storing successive letter counts
bigrams = Counter(zip(f_bytes, f_bytes[1:]))

# Reverse sort bigrams tallied by Counter's dictionary item value,
# so the bigrams with the highest frequency appear first
sorted_bigrams = sorted(bigrams.items(), key=lambda v: v[1], reverse=True)

# Print the top 10 most frequently occurring bigrams in text file
num_bigrams = sum(bigrams.values())
for k, v in sorted_bigrams[:10]:  # For each item: k = key, v = value
    # Convert each key's value (a tuple of two ASCII values) to a string
    s = "".join(map(chr, k))
    print(f'Bigram {k}, "{s}": freq = {v / num_bigrams:>5.2%}')
