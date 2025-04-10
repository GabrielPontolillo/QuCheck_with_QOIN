from qiskit import QuantumCircuit


def superdense_coding(b1, b2):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    if b1 == 1:
        qc.z(0)
        qc.s(0)
        qc.s(0)
        qc.s(0)
        qc.s(0)

    if b2 == 1:
        qc.x(0)

    qc.cx(0, 1)
    qc.h(0)
    return qc
