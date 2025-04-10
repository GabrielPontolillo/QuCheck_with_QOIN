# class that inherits from property based test
import numpy as np
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators import RandomFourierTransformState
from case_studies.quantum_fourier_transform.quantum_fourier_transform import quantum_fourier_transform


class PhaseShiftToLinearShift(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        state = RandomFourierTransformState(2, 5)
        return [state]

    # specify the preconditions for the test
    def preconditions(self, state):
        return True

    # specify the operations to be performed on the input
    def operations(self, state):
        n = state.num_qubits

        qft_1 = QuantumCircuit(n, n)
        qft_1.initialize(state, range(n))
        qft_1 = qft_1.compose(quantum_fourier_transform(n, swap=True).inverse(), reversed(range(n)))
        qft_1 = linear_shift(qft_1)

        qft_2 = QuantumCircuit(n, n)
        qft_2.initialize(state, range(n))
        qft_2 = phase_shift(qft_2)
        qft_2 = qft_2.compose(quantum_fourier_transform(n, swap=True).inverse(), reversed(range(n)))

        self.statistical_analysis.assert_equal(self, list(range(n)), qft_1, list(range(n)), qft_2)

def phase_shift(qc):
    # we instead apply a positive phase shift to the qubits as the linear shift adds 1 (down shift instead of up)
    for i in range(qc.num_qubits):
        qc.p(np.pi / 2 ** (qc.num_qubits - 1 - i), i)
    return qc


# Apply a series of controlled gates to perform the addition of 1
def linear_shift(qc):
    # this
    n = qc.num_qubits
    for i in reversed(range(1, n)):
        qubit_list = [j for j in range(i)]
        qc.mcx(qubit_list, i)
    qc.x(0)

    return qc