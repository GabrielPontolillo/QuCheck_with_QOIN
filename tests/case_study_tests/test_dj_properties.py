import unittest

from case_studies.deutsch_jozsa.balanced_property import DeutschJozsaWorksForBalancedFunction
from case_studies.deutsch_jozsa.constant_property import DeutschJozsaWorksForConstantFunction
from case_studies.deutsch_jozsa.dj_lower_register_minus_property import DeutschJozsaLowerRegisterMinus
from case_studies.deutsch_jozsa.vmerge_two_balanced_oracles_property import DeutschJozsaVMergeTwoBalancedOracles
from case_studies.deutsch_jozsa.vmerge_two_constant_oracles_property import DeutschJozsaVMergeTwoConstantOracles
from qucheck.test_runner import TestRunner


class TestDeutschJozsa(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def test_balanced_properties(self):
        # run the test
        runner = TestRunner([DeutschJozsaWorksForBalancedFunction], self.num_inputs, 42, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [DeutschJozsaWorksForBalancedFunction]

    def test_constant_properties(self):
        # run the test
        runner = TestRunner([DeutschJozsaWorksForConstantFunction], self.num_inputs, 4, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [DeutschJozsaWorksForConstantFunction]

    def test_vmerge_two_constant_oracles(self):
        # run the test
        runner = TestRunner([DeutschJozsaVMergeTwoConstantOracles], self.num_inputs, 3, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [DeutschJozsaVMergeTwoConstantOracles]

    def test_vmerge_two_balanced_oracles(self):
        # run the test
        runner = TestRunner([DeutschJozsaVMergeTwoBalancedOracles], self.num_inputs, 2, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [DeutschJozsaVMergeTwoBalancedOracles]

    def test_lower_register_minus(self):
        # run the test
        runner = TestRunner([DeutschJozsaLowerRegisterMinus], self.num_inputs, 11, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [DeutschJozsaLowerRegisterMinus]
