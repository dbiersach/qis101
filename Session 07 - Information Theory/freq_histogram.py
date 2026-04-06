#!/usr/bin/env -S uv run
"""freq_histogram.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = "gettysburg.txt"
FILE_NAME = "ciphertext1.txt"

# Read the data file into a buffer
file_path = Path(__file__).parent / FILE_NAME
with open(file_path, "rb") as f_in:
    f_bytes = bytearray(f_in.read())

# Only set tick marks for characters that occur more than 6%
char_count = np.bincount(np.asarray(f_bytes, dtype=np.uint8), minlength=256)
ticks = [char for char, count in enumerate(char_count) if count / len(f_bytes) > 0.06]

# Create a histogram of each character's ASCII value
plt.figure(Path(__file__).name)
plt.bar(np.arange(256), char_count)
plt.xticks(ticks)
plt.tick_params("x", rotation=90)
plt.title(f"Frequency Analysis ({FILE_NAME})")
plt.xlabel("ASCII Value")
plt.ylabel("Count")
plt.show()
