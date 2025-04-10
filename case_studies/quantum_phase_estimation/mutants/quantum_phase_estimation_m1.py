import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def quantum_phase_estimation(estimation_qubits, unitary_gate, eigenstate_of_unitary):
    qpe = QuantumCircuit(estimation_qubits + unitary_gate.num_qubits, estimation_qubits + unitary_gate.num_qubits)

    # initialise the unitary register to the specified eigenvector of U
    qpe.initialize(eigenstate_of_unitary, list(range(estimation_qubits, estimation_qubits + unitary_gate.num_qubits)))

    # apply hadamard to all qubits in estimating register
    for i in range(estimation_qubits):
        qpe.s(i)  # here

    unitary_qubits = [i + estimation_qubits for i in range(unitary_gate.num_qubits)]

    # do controlled unitary operations
    for i in range(estimation_qubits):
        for j in range(2 ** i):
            qubits = [estimation_qubits - 1 - i]
            qubits.extend(unitary_qubits)
            qpe = qpe.compose(unitary_gate.control(), qubits)

    qpe = qpe.compose(qft_general(estimation_qubits).inverse(), range(estimation_qubits))

    return qpe


def qft_general(qubits, swap=True):
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
            qft.swap(qubit, qubits - 1 - qubit)
    return qft
