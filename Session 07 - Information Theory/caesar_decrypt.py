#!/usr/bin/env -S uv run
"""caesar_decrypt.py"""

from pathlib import Path

FILE_NAME = "ciphertext1.txt"
KEY_SHIFT = 0

# Read the file as bytes
file_path = Path(__file__).parent / FILE_NAME
with open(file_path, "rb") as f_in:
    f_bytes = bytearray(f_in.read())

# Apply the Caesar cipher shift
f_bytes = bytearray((b + KEY_SHIFT) % 256 for b in f_bytes)

# Decode file bytes to UTF-8 and display the result
print(f_bytes.decode("utf-8", "ignore"))
