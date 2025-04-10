import unittest

from case_studies.superdense_coding.correct_bits_sent_property import CorrectBitsSent
from tests.mock_properties.other_mocks import IncorrectBitsSent

from qucheck.test_runner import TestRunner


# test the identity property
class TestQTProperties(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def test_equal_input_output(self):
        # run the test
        runner = TestRunner([CorrectBitsSent], self.num_inputs, 1, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [CorrectBitsSent]

    def test_equal_input_output_fails_also(self):
        # run the test
        runner = TestRunner([IncorrectBitsSent], self.num_inputs, 1, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == [IncorrectBitsSent]
        assert runner.list_passing_properties() == []
