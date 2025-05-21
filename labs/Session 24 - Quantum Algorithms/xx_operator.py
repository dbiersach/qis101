'''
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# --- State Preparation ---
# Start in |00> by default.
# Apply X to qubit 1 to prepare |0>|1>
qc.x(1)

# Apply Hadamard on qubit 0: creates (|0> + |1>)/sqrt(2) ⊗ |1>
qc.h(0)

# Entangle qubits with CNOT (control: qubit 0, target: qubit 1)
# This yields (|01> + |10>)/sqrt(2)
qc.cx(0, 1)

# Apply Z gate on qubit 1 to flip the phase of |01> so that
# the state becomes (|01> - |10>)/sqrt(2)
qc.z(1)

# (Optional) Check the statevector before applying XX
sv_before = Statevector.from_instruction(qc)
print("Statevector before applying XX:")
print(sv_before)

# --- Applying the XX Operator ---
# Applying an X gate on both qubits implements the XX operator:
qc.x(0)
qc.x(1)

# (Optional) Check the statevector after applying XX.
# Since the state is an eigenstate of XX (with eigenvalue -1), it picks up a global phase.
sv_after = Statevector.from_instruction(qc)
print("\nStatevector after applying XX:")
print(sv_after)

# --- Measurement ---
# Add measurement to the circuit
qc.measure(0, 0)
qc.measure(1, 1)

# Use the AerSimulator
sim = AerSimulator()

# Execute the circuit with 1024 shots
result = sim.run(qc, shots=1024).result()
counts = result.get_counts()
print("\nMeasurement counts after applying XX:")
print(counts)

# --- Expectation Value Calculation ---
# Calculate the expectation value of XX using the statevector (ignoring measurement)
# Define the Pauli-X matrix
X = np.array([[0, 1], [1, 0]])
# Construct the XX operator using the Kronecker product
XX = np.kron(X, X)

# Get the statevector data after applying XX (before measurement)
psi = sv_after.data  # numpy array representation
expectation_value = np.vdot(psi, XX @ psi)
print("\nExpectation value of XX:")
print(expectation_value)
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import plot_state_city

# Prepare lists to store statevectors and descriptions for each step.
states = []
descriptions = []

# Step 0: Initial state |00>
qc0 = QuantumCircuit(2)
state0 = Statevector.from_instruction(qc0)
states.append(state0)
descriptions.append("Initial |00>")

# Step 1: Apply X on qubit 1 -> state becomes |0⟩⊗|1⟩ = |01>
qc1 = QuantumCircuit(2)
qc1.x(1)
state1 = Statevector.from_instruction(qc1)
states.append(state1)
descriptions.append("After X on qubit 1: |01>")

# Step 2: Apply H on qubit 0 -> Qubit 0 becomes (|0⟩+|1⟩)/√2; Qubit 1 remains |1>
qc2 = QuantumCircuit(2)
qc2.x(1)
qc2.h(0)
state2 = Statevector.from_instruction(qc2)
states.append(state2)
descriptions.append("After H on qubit 0: Q0 in (|0⟩+|1⟩)/√2, Q1 is |1>")

# Step 3: Apply CX (control: qubit 0, target: qubit 1)
# State becomes (|01⟩+|10⟩)/√2 (entangled state)
qc3 = QuantumCircuit(2)
qc3.x(1)
qc3.h(0)
qc3.cx(0, 1)
state3 = Statevector.from_instruction(qc3)
states.append(state3)
descriptions.append("After CX: Entangled state (|01⟩+|10⟩)/√2")

# Step 4: Apply Z on qubit 1 -> state becomes (|01⟩-|10⟩)/√2 (singlet state)
qc4 = QuantumCircuit(2)
qc4.x(1)
qc4.h(0)
qc4.cx(0, 1)
qc4.z(1)
state4 = Statevector.from_instruction(qc4)
states.append(state4)
descriptions.append("After Z on qubit 1: Singlet state (|01⟩-|10⟩)/√2")

# For each step, compute the reduced density matrix for each qubit and plot using the city visualization.
for idx, (state, desc) in enumerate(zip(states, descriptions)):
    # Reduced density matrix for qubit 0 (trace out qubit 1)
    rho0 = partial_trace(state, [1]).data
    # Reduced density matrix for qubit 1 (trace out qubit 0)
    rho1 = partial_trace(state, [0]).data

    # Plot for Qubit 0
    print(f"Step {idx}: Qubit 0 - {desc}")
    plot_state_city(rho0, title=f"Step {idx}: Qubit 0\n{desc}")

    # Plot for Qubit 1
    print(f"Step {idx}: Qubit 1 - {desc}")
    plot_state_city(rho1, title=f"Step {idx}: Qubit 1\n{desc}")

plt.show()
'''

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate

# Create a 2-qubit circuit
qc = QuantumCircuit(2)

# Append the RXXGate with theta = pi (equivalent to XX up to a global phase)
qc.append(RXXGate(np.pi), [0, 1])

# Draw the circuit
print(qc.draw())
