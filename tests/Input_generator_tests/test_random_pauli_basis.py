import itertools
import random

import numpy as np
import unittest

from functools import reduce
from qucheck.input_generators import RandomPauliBasisState
from qiskit.quantum_info import Statevector


class TestRandomPauliBasisState(unittest.TestCase):
    def setUp(self):
        self.generator = RandomPauliBasisState(2,  5)

    def test_generate_size(self):
        statevector = self.generator.generate(1)
        self.assertTrue(
            2 ** self.generator.number_of_qubits_low <= len(statevector) <= 2 ** self.generator.number_of_qubits_high)
        self.assertTrue(check_tensor_product_representation(statevector))

    def test_generate_different(self):
        statevector1 = self.generator.generate(1)
        statevector2 = self.generator.generate(2)
        self.assertFalse(np.array_equal(statevector1, statevector2))
        self.assertTrue(check_tensor_product_representation(statevector1))
        self.assertTrue(check_tensor_product_representation(statevector2))

    def test_generate_same_seed(self):
        statevector1 = self.generator.generate(337)
        statevector2 = self.generator.generate(337)
        self.assertTrue(np.array_equal(statevector1, statevector2))
        self.assertTrue(check_tensor_product_representation(statevector1))
        self.assertTrue(check_tensor_product_representation(statevector2))

    def test_generate_different_method_same_seed(self):
        statevector1 = self.generator.generate(337)
        statevector2 = RandomPauliBasisState(2, 5).generate(337)
        self.assertTrue(np.array_equal(statevector1, statevector2))
        self.assertTrue(check_tensor_product_representation(statevector1))
        self.assertTrue(check_tensor_product_representation(statevector2))

    def test_generate_normalized(self):
        statevector = self.generator.generate(1)
        self.assertTrue(Statevector(statevector).is_valid())
        self.assertTrue(check_tensor_product_representation(statevector))

    def test_generate_tensor_product(self):
        statevector = self.generator.generate(random.randint(0, 100000))
        # Check if the statevector can be represented as a tensor product of the |+>,|->,|0>,|1>,|i>,|-i> states
        # This test will depend on your implementation of the tensor product representation check
        self.assertTrue(check_tensor_product_representation(statevector))


def check_tensor_product_representation(statevector):
    # Define the six basis states
    basis_states = [Statevector.from_label('+').data,
                    Statevector.from_label('-').data,
                    Statevector.from_label('0').data,
                    Statevector.from_label('1').data,
                    Statevector.from_label('r').data,
                    Statevector.from_label('l').data]

    # Calculate the number of qubits from the length of the statevector
    number_of_qubits = int(np.log2(len(statevector)))

    # Loop over all possible combinations of these states
    for combination in itertools.product(basis_states, repeat=number_of_qubits):
        # Calculate the tensor product of the combination
        tensor_product = reduce(np.kron, combination)
        # Compare it to the input statevector
        if np.allclose(tensor_product, statevector):
            return True

    # If no match is found after checking all combinations, return False
    return False
