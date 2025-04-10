import itertools
import numpy as np
import unittest

from case_studies.deutsch_jozsa.dj_helpers import BalancedOracleInputGenerator, ConstantOracleInputGenerator, RandomOracleInputGenerator
from qiskit.quantum_info import Statevector, Operator


class TestDJOracles(unittest.TestCase):
    def test_balanced_generate_size(self):
        oracle = BalancedOracleInputGenerator(2, 5).generate(1)
        print(oracle)
        self.assertEqual(oracle.num_qubits, 3)

    def test_balanced_generate_different(self):
        oracle1 = Operator.from_circuit(BalancedOracleInputGenerator(2, 5).generate(1))
        oracle2 = Operator.from_circuit(BalancedOracleInputGenerator(2, 5).generate(2))

        print(oracle1.to_matrix())
        print(oracle2.to_matrix())

        # convert the quantum circuits to matrices and compare them
        self.assertFalse(np.array_equal(oracle1.to_matrix(), oracle2.to_matrix()))

    def test_balanced_generate_same_seed(self):
        oracle1 = Operator.from_circuit(BalancedOracleInputGenerator(2, 5).generate(1060))
        oracle2 = Operator.from_circuit(BalancedOracleInputGenerator(2, 5).generate(1060))

        print(oracle1.to_matrix())
        print(oracle2.to_matrix())

        # convert the quantum circuits to matrices and compare them
        self.assertTrue(np.array_equal(oracle1.to_matrix(), oracle2.to_matrix()))

    def test_constant_generate_size(self):
        oracle = ConstantOracleInputGenerator(2, 5).generate(1)
        print(oracle)
        self.assertEqual(oracle.num_qubits, 3)

    def test_constant_generate_different(self):
        oracle1 = Operator.from_circuit(ConstantOracleInputGenerator(2, 5).generate(1))
        oracle2 = Operator.from_circuit(ConstantOracleInputGenerator(2, 5).generate(2))

        print(oracle1.to_matrix())
        print(oracle2.to_matrix())

        # convert the quantum circuits to matrices and compare them
        self.assertFalse(np.array_equal(oracle1.to_matrix(), oracle2.to_matrix()))

    def test_constant_generate_same_seed(self):
        oracle1 = Operator.from_circuit(ConstantOracleInputGenerator(2, 5).generate(10320))
        oracle2 = Operator.from_circuit(ConstantOracleInputGenerator(2, 5).generate(10320))

        print(oracle1.to_matrix())
        print(oracle2.to_matrix())

        # convert the quantum circuits to matrices and compare them
        self.assertTrue(np.array_equal(oracle1.to_matrix(), oracle2.to_matrix()))

    def test_random_generate_size(self):
        oracle = RandomOracleInputGenerator(2, 5).generate(1)
        print(oracle)
        self.assertEqual(oracle.num_qubits, 3)

    def test_random_generate_different(self):
        oracle1 = Operator.from_circuit(RandomOracleInputGenerator(2, 5).generate(1))
        oracle2 = Operator.from_circuit(RandomOracleInputGenerator(2, 5).generate(2))

        print(oracle1.to_matrix())
        print(oracle2.to_matrix())

        # convert the quantum circuits to matrices and compare them
        self.assertFalse(np.array_equal(oracle1.to_matrix(), oracle2.to_matrix()))

    def test_random_generate_same_seed(self):
        oracle1 = Operator.from_circuit(RandomOracleInputGenerator(2, 5).generate(10320))
        oracle2 = Operator.from_circuit(RandomOracleInputGenerator(2, 5).generate(10320))

        print(oracle1.to_matrix())
        print(oracle2.to_matrix())

        # convert the quantum circuits to matrices and compare them
        self.assertTrue(np.array_equal(oracle1.to_matrix(), oracle2.to_matrix()))
