#!/usr/bin/env python3
"""make_zeta_zeros.py"""

import lzma
import pickle
from pathlib import Path

import numpy as np

# Import the list of the first 10,000 zeros of the Riemann Zeta function
# Special thanks to Andrew Odlyzko and AT&T for providing these values
# See https://www-users.cse.umn.edu/~odlyzko/zeta_tables/index.html
file_path = Path(__file__).parent / "zeta_zeros.txt"
zeta_zeros = np.genfromtxt(file_path)

# Write pickle file of 9-digit zeta zeros
file_path = Path(__file__).parent / "zeta_zeros.pickle.xz"
with lzma.open(file_path, "wb") as file_out:
    pickle.dump(zeta_zeros, file_out, pickle.HIGHEST_PROTOCOL)

print(f"Created {file_path}")
