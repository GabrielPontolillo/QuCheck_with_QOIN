import numpy as np
from qiskit.quantum_info import Operator, Statevector

from qucheck.input_generators.input_generator import InputGenerator

from typing import Tuple, Any


# This implementation is expensive, if using large circuits look to implement a less random implementation
class RandomEigenvectorUnitaryPair(InputGenerator):
    def __init__(self, unitary_generator: InputGenerator, number_of_eigenvectors_to_return=None):
        self.number_of_eigenvectors_to_return = number_of_eigenvectors_to_return

        # genrator should return a Qiskit.quantum_info.Operator, as we call .data to get the matrix
        # TODO: generalise this to allow for a generator that returns other types of objects that can be converted to a matrix
        self.unitary_generator = unitary_generator

    # returns a tuple of n linearly independent eigenvectors and their corresponding eigenvalues,
    # with the generated unitary
    def generate(self, seed) -> Tuple[Tuple[Tuple[Statevector, Any], ...], Operator]:
        # Will throw an error if number of eigenvectors to return is greater than the number of linearly independent eigenvectors 2^#qubits
        unitary = self.unitary_generator.generate(seed)

        # get all eigenvector, eigenvalue pairs
        if self.number_of_eigenvectors_to_return is None:
            self.number_of_eigenvectors_to_return = 2**unitary.num_qubits

        # generate a random eigenvector
        # TODO: this will break if the input generator passed does not generate a Qiskit.quantum_info.Operator
        eigenvalues, eigenvectors = np.linalg.eig(unitary.data)

        # choose n random eigenvectors
        indices = np.random.choice(len(eigenvalues), size=self.number_of_eigenvectors_to_return, replace=False)

        # build output pair tuple
        eigenvector_vector_value_pairs: Tuple[Tuple[Statevector, Any], ...] = tuple(
            (Statevector(eigenvectors[:, i]), eigenvalues[i])
            for i in indices
        )

        return eigenvector_vector_value_pairs, unitary
