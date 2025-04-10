import unittest

from case_studies.quantum_teleportation.input_reg0_equal_to_output_reg2_property import Inq0EqualOutq2
from case_studies.quantum_teleportation.not_teleported_registers_equal_to_plus_property import NotTeleportedPlus
from case_studies.quantum_teleportation.unitary_before_teleport_equal_unitary_after_teleport_property import UnitaryBeforeAndAfterTeleport
from qucheck.test_runner import TestRunner


# test the identity property
class TestQTProperties(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def test_equal_input_output(self):
        # run the test
        runner = TestRunner([Inq0EqualOutq2], self.num_inputs, 1, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [Inq0EqualOutq2]

    def test_not_teleported_plus(self):
        # run the test
        runner = TestRunner([NotTeleportedPlus], self.num_inputs, 1, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [NotTeleportedPlus]

    def test_unitary_before_and_after_teleport(self):
        # run the test
        runner = TestRunner([UnitaryBeforeAndAfterTeleport], self.num_inputs, 1, self.num_measurements)
        runner.run_tests()
        # the property should pass
        assert runner.list_failing_properties() == []
        assert runner.list_passing_properties() == [UnitaryBeforeAndAfterTeleport]
