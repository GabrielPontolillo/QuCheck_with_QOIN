from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


def superdense_coding(b1, b2):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    if b1 == 1:
        qc.z(0)

    qc.z(0)
    qc.y(0)
    qc.append(Pauli('-iX'), [1])

    if b2 == 1:
        qc.x(0)

    qc.cx(0, 1)
    qc.h(0)
    return qc
