from qiskit import QuantumCircuit


# returns the quantum_teleportation circuit
def quantum_teleportation():
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)


    qc.cx(1, 2)
    __qmutpy_qgi_func__(qc, 0, 2)
    return qc


def __qmutpy_qgi_func__(circ, arg1, arg2):
    circ.cz(arg1, arg2)
    circ.cz(arg1, arg2)
