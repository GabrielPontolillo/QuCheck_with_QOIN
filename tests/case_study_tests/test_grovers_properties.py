import unittest

from case_studies.grovers_algorithm.grover_lower_register_minus_property import GroversAlgorithmLowerRegisterMinus
from case_studies.grovers_algorithm.most_frequent_output_should_not_be_marked_when_too_many_marks import GroversAlgorithmMostFrequentNotMarkedIfTooManyMarked
from case_studies.grovers_algorithm.most_frequent_should_be_marked import GroversAlgorithmMostFrequentMarked
from qucheck.test_runner import TestRunner


class TestGrovers(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def test_lower_register_minus(self):
        # run the test
        runner = TestRunner([GroversAlgorithmLowerRegisterMinus], self.num_inputs, 42, self.num_measurements)
        print("GroversAlgorithmLowerRegisterMinus")
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [GroversAlgorithmLowerRegisterMinus]

    def test_most_frequent_marked(self):
        # run the test
        runner = TestRunner([GroversAlgorithmMostFrequentMarked], self.num_inputs, 41, self.num_measurements)
        print("GroversAlgorithmMostFrequentMarked")
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [GroversAlgorithmMostFrequentMarked]

    def test_most_frequent_not_marked_when_too_many_marks(self):
        # run the test
        runner = TestRunner([GroversAlgorithmMostFrequentNotMarkedIfTooManyMarked], self.num_inputs, 40, self.num_measurements)
        print("GroversAlgorithmMostFrequentNotMarkedIfTooManyMarked")
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [GroversAlgorithmMostFrequentNotMarkedIfTooManyMarked]
