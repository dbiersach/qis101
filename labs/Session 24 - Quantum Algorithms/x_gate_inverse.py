import numpy as np
from qiskit.circuit.library import XGate
from qiskit.quantum_info import Operator

# Create an X gate instance
x_gate = XGate()

# Obtain the unitary matrix for the X gate
x_matrix = Operator(x_gate).data
print("X gate matrix:\n", x_matrix)

# Compute the inverse of the X gate's matrix
x_inv_matrix = np.linalg.inv(x_matrix)
print("\nInverse of X gate matrix:\n", x_inv_matrix)

# Verify that the inverse is the same as the original matrix
is_self_inverse = np.allclose(x_matrix, x_inv_matrix)
print("\nIs the X gate its own inverse? ", is_self_inverse)
