from qiskit import QuantumCircuit


def superdense_coding(b1, b2):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    if b1 == 1:
        qc.z(0)
    if b2 == 1:
        qc.x(0)

    __qmutpy_qgi_func__(qc, 0, 1)
    qc.h(0)
    return qc


def __qmutpy_qgi_func__(circ, arg1, arg2):
     circ.cx(arg1, arg2)
     circ.dcx(arg1, arg2)
