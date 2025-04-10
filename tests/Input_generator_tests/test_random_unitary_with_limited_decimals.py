import numpy as np
import unittest

from qucheck.input_generators import RandomUnitaryLimitedDecimals


class TestRandomUnitaryLimitedDecimals(unittest.TestCase):
    def setUp(self):
        self.generator = RandomUnitaryLimitedDecimals(1, 5, 3)

    def test_generate_size(self):
        unitary = self.generator.generate(1).data
        print(unitary)

    def test_generate_different(self):
        unitary1 = self.generator.generate(1).data
        unitary2 = self.generator.generate(4).data
        self.assertFalse(np.array_equal(unitary1, unitary2))

    def test_generate_same_seed(self):
        unitary1 = self.generator.generate(337).data
        unitary2 = self.generator.generate(337).data
        self.assertTrue(np.array_equal(unitary1, unitary2))






