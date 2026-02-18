#!/usr/bin/env -S uv run
"""rsa_demo.py"""


def extended_euclidean(a, b):
    swapped = False
    if a < b:
        a, b = b, a
        swapped = True
    ca, cb = ((1, 0), (0, 1))
    while b != 0:
        k = int(a // b)
        a, b, ca, cb = (b, a - b * k, (cb), (ca[0] - k * cb[0], ca[1] - k * cb[1]))
    if swapped:
        return ca[1], ca[0]
    else:
        return ca


def power_modulus(base, exponent, modulus):
    r = 1
    for i in range(exponent.bit_length(), -1, -1):
        r = (r * r) % modulus
        if (exponent >> i) & 1:
            r = (r * base) % modulus
    return r


def generate_keys(p, q):
    # Calculate the public modulus (product of two primes)
    n = p * q
    # As p and q are both primes, Euler's totient is simple
    totient = (p - 1) * (q - 1)
    # Set PUBLIC encryption exponent (a random prime number)
    e = 35537
    # Calculate the corresponding PRIVATE encryption exponent
    d = extended_euclidean(e, totient)[0]
    if d < 0:
        d += totient
    return n, e, d


def main():
    # Pick two (normally large random) prime numbers
    p, q = 31337, 31357

    n, e, d = generate_keys(p, q)
    print(f"Public RSA Modulus = {n:,}")
    print(f"Public Encryption Key = {e:,}")
    print(f"Private Decryption Key = {d:,}")
    print()

    plaintext = "Hi!"
    print(f"Plaintext = {plaintext}")

    b = bytearray(plaintext, encoding="utf-8")
    print("Plaintext as byte array = ", end="")
    print(*b, sep=", ")
    plaintext_int = int.from_bytes(b)
    print(f"Plaintext as an Integer = {plaintext_int:,}")
    print()

    ciphertext_int = power_modulus(plaintext_int, e, n)
    print(f"Ciphertext as an Integer = {ciphertext_int:,}")
    print()

    plaintext_int = power_modulus(ciphertext_int, d, n)
    print(f"Plaintext as an Integer = {plaintext_int:,}")

    plaintext_length = (plaintext_int.bit_length() + 7) // 8
    b = bytearray(plaintext_int.to_bytes(plaintext_length))
    plaintext = b.decode(encoding="utf-8")
    print(f"Plaintext = {plaintext}")


if __name__ == "__main__":
    main()
