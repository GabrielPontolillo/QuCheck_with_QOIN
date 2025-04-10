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
            qft.cp(rotation_amount, control_index, target_index)

    # do swaps
    if swap:
        for qubit in range(qubits // 2):
            qft.p(qubit, qubits - 1 - qubit)  # here
    return qft
