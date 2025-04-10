import random
import numpy as np

from qiskit.quantum_info import Operator

from qucheck.input_generators.input_generator import InputGenerator


# add input generator for this specific scenario
class RandomUnitaryLimitedDecimals(InputGenerator):
    # choose the fraction of 2pi to limit the decimals to
    def __init__(self, number_of_qubits_low, number_of_qubits_high, binary_fraction):
        self.number_of_qubits_low = number_of_qubits_low
        self.number_of_qubits_high = number_of_qubits_high
        self.binary_fraction = binary_fraction

    def generate(self, seed) -> Operator:
        random.seed(seed)
        np.random.seed(seed)

        number_of_qubits = random.randint(self.number_of_qubits_low, self.number_of_qubits_high)

        eigenvalues = [np.exp(2j * np.pi * k / (2 ** self.binary_fraction)) for k in np.random.randint(0, 2 ** self.binary_fraction, 2 ** number_of_qubits)]

        unitary_matrix = np.diag(eigenvalues)

        return Operator(unitary_matrix)
