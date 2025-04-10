import numpy as np
import unittest

from qucheck.input_generators import RandomState
from qiskit.quantum_info import Statevector


class TestRandomState(unittest.TestCase):
    def setUp(self):
        self.generator = RandomState(2)

    def test_generate_size(self):
        statevector = self.generator.generate(1)
        self.assertEqual(len(statevector), 2 ** self.generator.number_of_qubits)

    def test_generate_different(self):
        statevector1 = self.generator.generate(1)
        statevector2 = self.generator.generate(2)
        self.assertFalse(np.array_equal(statevector1, statevector2))

    def test_generate_same_seed(self):
        statevector1 = self.generator.generate(337)
        statevector2 = self.generator.generate(337)
        self.assertTrue(np.array_equal(statevector1, statevector2))

    def test_generate_normalized(self):
        statevector = self.generator.generate(1)
        self.assertTrue(Statevector(statevector).is_valid())
