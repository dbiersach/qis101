#!/usr/bin/env -S uv run
"""bigram_decrypt.py"""

from pathlib import Path

FILE_NAME = "bigram_ciphertext.txt"

# Read the file as bytes
file_path = Path(__file__).parent / FILE_NAME
with open(file_path, "rb") as f_in:
    f_bytes = bytearray(f_in.read())

# Decrypt the file by applying a bitwise XOR operation with 128 to each byte
f_bytes = bytearray(b ^ 128 for b in f_bytes)

# Display the first 500 characters in the decrypted file
print(f_bytes.decode()[:500])
