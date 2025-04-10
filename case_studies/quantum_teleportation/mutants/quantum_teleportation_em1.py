from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


# returns the quantum_teleportation circuit
def quantum_teleportation():
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)

    qc.cx(1, 2)
    qc.cz(0, 2)

    # semantic preserving changes loc 7, gates 4, index 0
    qc.y(0)
    qc.y(0)

    return qc

