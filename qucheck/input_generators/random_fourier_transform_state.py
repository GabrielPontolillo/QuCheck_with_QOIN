# class for generating random statevectors using Qiskit, the generator method receives the dimensions of the statevector
import random
from cmath import exp
import numpy as np

from qiskit.quantum_info import Statevector
from qucheck.input_generators.input_generator import InputGenerator


class RandomFourierTransformState(InputGenerator):
    def __init__(self, number_of_qubits_low, number_of_qubits_high):
        self.number_of_qubits_low = number_of_qubits_low
        self.number_of_qubits_high = number_of_qubits_high

    def generate(self, seed) -> Statevector:
        random.seed(seed)

        # need to return a statevector that would be returned by performing QFT on a computational basis state
        # we need to consider the roots of unity matrix for a given size, as well as the column to select

        # select the number of qubits
        n = random.randint(self.number_of_qubits_low, self.number_of_qubits_high)

        # calculate w for the roots of unity matrix
        w = exp(2 * np.pi * 1j / (2 ** n))

        # generate a random number to select the column
        k = random.randint(0, (2 ** n) - 1)

        state = np.zeros(2 ** n, dtype=complex)

        for i in range(2 ** n):
            state[i] = w ** ((k * i) % 2 ** n)

        # multiply all elements in the state by 1/sqrt(2^n) to normalize the state
        state = state / np.sqrt(2 ** n)

        return Statevector(state)
