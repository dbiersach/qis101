#!/usr/bin/env python3
"""oscillating_integrand.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

a, b = 0, 5000
x = np.linspace(a, b, 1000)
plt.figure(Path(__file__).name)
plt.plot(x, np.cos(x))
plt.title(r"$f(x)=\cos{x}$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# scipy.quad()'s default limit of 50 subintervals may fail
# for oscillatory integrands evaluated over large intervals
first_estimate = quad(np.cos, a, b, full_output=True)[0]
print(f"{ first_estimate = :.6f}")

# We can improve scipy.quad()'s adaptive quadrature
# by specifying a greater subinterval limit
second_estimate = quad(np.cos, a, b, full_output=True, limit=1000)[0]
print(f"{second_estimate = :.6f}")

# Compare these estimates to the exact analytic integral
print(f"    F(b) - F(a) = {np.sin(5000):.6f}")
