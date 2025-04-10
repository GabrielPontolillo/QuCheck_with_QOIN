import random
from qiskit.quantum_info import random_unitary, Operator

from qucheck.input_generators.input_generator import InputGenerator


class RandomUnitary(InputGenerator):
    def __init__(self, number_of_qubits_low, number_of_qubits_high):
        self.number_of_qubits_low = number_of_qubits_low
        self.number_of_qubits_high = number_of_qubits_high

    def generate(self, seed) -> Operator:
        number_of_qubits = random.randint(self.number_of_qubits_low, self.number_of_qubits_high)
        # generate a random statevector
        return random_unitary(2 ** number_of_qubits, seed)
