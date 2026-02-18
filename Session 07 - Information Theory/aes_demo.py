#!/usr/bin/env -S uv run
"""aes_demo.py"""

import os

from aes_library import AES


def main():
    # The secret key is 16 bytes long
    secret_key = os.urandom(16)
    print(f"Secret key = {bytearray(secret_key).hex()}")

    # Create a (b) "bytes" object, not a string
    plaintext = b"Attack at dawn"
    print(f"{plaintext.decode('ascii')}")
    print("plaintext = ")
    print([f"0x{b:02x}" for b in bytearray(plaintext)], sep=", ")

    # The random initialization vector (iv) ensures the same value
    # encrypted multiple times, even with the same secret key,
    # will not always result in the same encrypted value
    # Note: The iv is sent along with the ciphertext to the receiver
    iv = os.urandom(16)

    ciphertext = AES(secret_key).encrypt_ctr(plaintext, iv)
    print("ciphertext =")
    print([f"0x{b:02x}" for b in bytearray(ciphertext)], sep=", ")

    plaintext = AES(secret_key).decrypt_ctr(ciphertext, iv)
    print("plaintext = ")
    print([f"0x{b:02x}" for b in bytearray(plaintext)], sep=", ")
    print(f"{plaintext.decode('ascii')}")


if __name__ == "__main__":
    main()
