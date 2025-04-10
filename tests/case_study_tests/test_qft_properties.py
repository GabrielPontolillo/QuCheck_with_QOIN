from case_studies.quantum_fourier_transform.identity_property import IdentityProperty
from case_studies.quantum_fourier_transform.linear_shift_induces_phase_shift_property import LinearShiftToPhaseShift
from case_studies.quantum_fourier_transform.phase_shift_induces_linear_shift_property import PhaseShiftToLinearShift
from qucheck.test_runner import TestRunner


import unittest


# test the identity property
class TestIdentityPropertyQFT(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def test_identity_property_qft(self):
        # run the test
        runner = TestRunner([IdentityProperty], self.num_inputs, 1, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [IdentityProperty]

    def test_phase_shift_property_qft(self):
        # run the test
        runner = TestRunner([LinearShiftToPhaseShift], self.num_inputs, 2, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [LinearShiftToPhaseShift]

    def test_linear_shift_property_qft(self):
        # run the test
        runner = TestRunner([PhaseShiftToLinearShift], self.num_inputs, 5, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [PhaseShiftToLinearShift]
