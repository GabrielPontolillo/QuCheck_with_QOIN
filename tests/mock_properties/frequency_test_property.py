# class that inherits from property based test
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators import Integer

import random


class FrequencyProperty(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        # The assumption of our code is that we need at least one input generator, if we do not include it,
        # the code breaks in the test runner - also its a fine assumption to include, property based testing requires inputs
        return [Integer(0, 255), Integer(0, 3)]

    # specify the preconditions for the test
    def preconditions(self, state, padding):
        return True

    # specify the operations to be performed on the input
    def operations(self, state, padding):
        padding = 2
        binary = bin(state)[2:]
        binary = '0' * padding + binary

        qc = QuantumCircuit(len(binary), len(binary))

        binary_to_mark(qc, binary)

        print(qc)

        self.statistical_analysis.assert_most_frequent(self, list(range(qc.num_qubits)), qc, [binary], basis=["z"])


class UncertainFrequencyProperty(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        # The assumption of our code is that we need at least one input generator, if we do not include it,
        # the code breaks in the test runner - also its a fine assumption to include, property based testing requires inputs
        return [Integer(0, 255), Integer(0, 3)]

    # specify the preconditions for the test
    def preconditions(self, state, padding):
        return True

    # specify the operations to be performed on the input
    def operations(self, state, padding):
        padding = 2
        binary = bin(state)[2:]
        binary = '0' * padding + binary

        qc = QuantumCircuit(len(binary), len(binary))

        binary_to_mark(qc, binary)

        # insert a random hadamard at a random position
        index = random.randint(0, len(binary) - 1)

        qc.h(index)

        # list with two possible outcomes after hadamard
        states = [binary[:index] + "0" + binary[index + 1:], binary[:index] + "1" + binary[index + 1:]]

        self.statistical_analysis.assert_most_frequent(self, list(range(qc.num_qubits)), qc, states, basis=["z"])


def binary_to_mark(qc: QuantumCircuit, binary_string: str):
    """
    Construct a circuit to mark a binary string.
    """
    for idx, bit in enumerate(binary_string):
        if bit == "1":
            qc.x(qc.qubits[idx])

