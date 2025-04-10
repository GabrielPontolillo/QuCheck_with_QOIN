import random
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.quantum_info import Operator

from qucheck.input_generators.input_generator import InputGenerator


class RandomTensorProductOfUnitary(InputGenerator):
    # TODO: Make this more general to receive a list of UnitaryGates and choose from those instead
    def __init__(self, number_of_qubits_low, number_of_qubits_high, unitaries=('X', 'Y', 'Z')):
        self.number_of_qubits_low = number_of_qubits_low
        self.number_of_qubits_high = number_of_qubits_high
        self.unitaries = unitaries

    def generate(self, seed) -> Operator:
        # print(seed)
        # print(random)
        random.seed(seed)

        # generate a random statevector
        pauli_string = ""

        for i in range(random.randint(self.number_of_qubits_low, self.number_of_qubits_high)):
            chosen_char = random.choice(self.unitaries)
            pauli_string += chosen_char

        # convert the string to a statevector
        return Operator(PauliGate(pauli_string).to_matrix())