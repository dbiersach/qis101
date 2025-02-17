# element_wise.py

import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"{x = }")

a = x > 2
print(f"{a = }")

b = x < 8
print(f"{b = }")

c = a & b
print(f"{c = }")
print(f"{x[c] = }")

print(x[np.where((x > 2) & (x < 8))])
