import numpy as np
from qiskit import QuantumCircuit


# returns the quantum fourier transform circuit
def quantum_fourier_transform(qubits, swap=True):
    # build circuit
    qft = QuantumCircuit(qubits, qubits)

    # modify phase
    for qubit in range(qubits):
        # insert the initial hadamard gate on all qubits in the register
        qft.h(qubit)

        # iterate across all indexes to get the appropriate controlled gates
        for offset in range(1, qubits - qubit):
            control_index = qubit + offset
            target_index = qubit
            rotation_amount = (np.pi / 2 ** offset)
            __qmutpy_qgi_func__(qft, rotation_amount, control_index, target_index)  # here

    # do swaps
    if swap:
        for qubit in range(qubits // 2):
            qft.swap(qubit, qubits - 1 - qubit)
    return qft


def __qmutpy_qgi_func__(circ, arg1, arg2, arg3):
    circ.cp(arg1, arg2, arg3)
    circ.cry(arg1, arg2, arg3)
