from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


# returns the quantum_teleportation circuit
def quantum_teleportation():
    qc = QuantumCircuit(3)

    # semantic preserving changes loc 1, gates 3, index 1
    qc.x(1)
    qc.x(1)

    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)

    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc

