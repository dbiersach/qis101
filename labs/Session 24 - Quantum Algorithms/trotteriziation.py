import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Total evolution time and number of Trotter steps
t = 1.0  # Total time
n_steps = 10  # Number of Trotter steps
dt = t / n_steps

# Create a quantum circuit with one qubit and one classical bit for measurement
qc = QuantumCircuit(1, 1)

# Trotterization for Hamiltonian H = X + Z:
# Approximate exp(-i H t) ≈ [exp(-i X dt) exp(-i Z dt)]^n_steps
for _ in range(n_steps):
    # RX gate implements exp(-i dt X) since RX(2*dt) = exp(-i dt X)
    qc.rx(2 * dt, 0)
    # RZ gate implements exp(-i dt Z) since RZ(2*dt) = exp(-i dt Z)
    qc.rz(2 * dt, 0)

# Add measurement to the circuit
qc.measure(0, 0)

# Plot and display the circuit diagram
circuit_fig = qc.draw("mpl")
plt.show()

# Set up the simulator
simulator = AerSimulator()

# Transpile the circuit for the simulator
compiled_circuit = transpile(qc, simulator)

# Run the circuit using the simulator; specifying shots (number of repetitions)
result = simulator.run(compiled_circuit, shots=1024).result()

# Retrieve and print the measurement counts
counts = result.get_counts()
print("Measurement counts:", counts)
