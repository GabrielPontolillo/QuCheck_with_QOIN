# class that inherits from property based test
import numpy as np
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators.random_pauli_basis_state import RandomPauliBasisState
from case_studies.quantum_fourier_transform.quantum_fourier_transform import quantum_fourier_transform


class LinearShiftToPhaseShift(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        state = RandomPauliBasisState(2, 5, tuple("z"))
        return [state]

    # specify the preconditions for the test
    def preconditions(self, state):
        return True

    # specify the operations to be performed on the input
    def operations(self, state):
        n = state.num_qubits

        qft_1 = QuantumCircuit(n, n)
        qft_1.initialize(state, reversed(range(n)))
        qft_1 = qft_1.compose(quantum_fourier_transform(n, swap=False))
        qft_1 = phase_shift(qft_1)

        init_state = state.data
        # make the first element the last element, and vice versa
        shifted_vector = np.roll(init_state, -1)

        qft_2 = QuantumCircuit(n, n)
        qft_2.initialize(shifted_vector, reversed(range(n)))
        qft_2 = qft_2.compose(quantum_fourier_transform(n, swap=False))

        self.statistical_analysis.assert_equal(self, list(range(n)), qft_1, list(range(n)), qft_2)


def phase_shift(qc):
    for i in range(qc.num_qubits):
        qc.p(-np.pi / 2 ** (qc.num_qubits - 1 - i), i)
    return qc
