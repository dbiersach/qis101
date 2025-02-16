#!/usr/bin/env python3
"""bigram_decrypt.py"""

from pathlib import Path

file_name = "bigram_ciphertext.txt"
file_path = Path(__file__).parent / file_name
with open(file_path, "rb") as f_in:
    f_bytes = bytearray(f_in.read())

for i in range(0, len(f_bytes)):
    f_bytes[i] = f_bytes[i] ^ 128

# Display the first 500 characters in the decrypted file
print(f_bytes.decode()[:500])
