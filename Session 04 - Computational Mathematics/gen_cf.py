#!/usr/bin/env -S uv run
"""gen_cf.py"""

import math


def decode_cf(form):
    a0, b0, Ai, Bi, Ci, Di, Ei = form
    an, bn = a0, b0
    hn, kn = 0, 0
    b_1, h_1, k_1 = 1, 1, 0
    h_2, k_2 = 0, 1
    for n in range(1, 50):  # max terms
        hn = an * h_1 + b_1 * h_2
        kn = an * k_1 + b_1 * k_2
        b_1 = bn
        h_1, h_2 = hn, h_1
        k_1, k_2 = kn, k_1
        an = Di * n + Ei
        bn = Ai * n * n + Bi * n + Ci
    return hn / kn


def main():
    print("Euler's Generalized Continued Fraction for Pi")
    act = math.pi
    est = decode_cf((3, 1, 4, 4, 1, 0, 6))
    err = abs((act - est) / act)
    print(f"Est : {est}")
    print(f"Act : {act}")
    print(f"Err : {err:.14%}\n")

    print("Biersach's Generalized Continued Fraction for Pi")
    est = decode_cf((2, 8, 4, 8, 0, 4, 2))
    err = abs((act - est) / act)
    print(f"Est : {est}")
    print(f"Act : {act}")
    print(f"Err : {err:.14%}\n")


if __name__ == "__main__":
    main()
