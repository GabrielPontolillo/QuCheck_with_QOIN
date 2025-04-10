# class for generating random statevectors using Qiskit, the generator method receives the dimensions of the statevector
from qucheck.input_generators.input_generator import InputGenerator
from qiskit.quantum_info import Statevector
import random


class RandomPauliBasisState(InputGenerator):
    def __init__(self, number_of_qubits_low, number_of_qubits_high, basis=('x', 'y', 'z')):
        self.number_of_qubits_low = number_of_qubits_low
        self.number_of_qubits_high = number_of_qubits_high
        self.basis = basis

    def generate(self, seed) -> Statevector:
        # generate a random statevector from the tensor product of the |+>,|->,|0>,|1>,|i>,|-i>,
        # loop through the number of qubits, choose one of the 6 statevectors
        state_string = ""

        state_list = []
        if 'x' in self.basis:
            state_list.extend(["+", "-"])
        if 'y' in self.basis:
            state_list.extend(["r", "l"])
        if 'z' in self.basis:
            state_list.extend(["0", "1"])

        random.seed(seed)
        for i in range(random.randint(self.number_of_qubits_low, self.number_of_qubits_high)):
            chosen_char = random.choice(state_list)
            state_string += chosen_char

        # convert the string to a statevector
        return Statevector.from_label(state_string)
