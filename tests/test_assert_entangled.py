# a test script for the test runner
from unittest import TestCase

from qucheck.test_runner import TestRunner
from tests.mock_properties.entangled_test_property import EntangledPrecondition, EntangledCheckOnUnentangledState, EntangledCheckOnGHZState, EntangledCheckOnRandomState


class TestAssertEntangled(TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def tearDown(self):
        TestRunner.property_objects = []
        TestRunner.seeds_list_dict = []

    def test_entangled_precondition(self):
        # create an instance of the test runner
        test_runner = TestRunner([EntangledPrecondition], self.num_inputs, 548, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        # list the failing properties
        assert test_runner.list_passing_properties() == [EntangledPrecondition]
        assert test_runner.list_failing_properties() == []

    def test_entangled_check_on_unentangled_state(self):
        # create an instance of the test runner
        test_runner = TestRunner([EntangledCheckOnUnentangledState], self.num_inputs, 548, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        # list the failing properties
        assert test_runner.list_passing_properties() == []
        assert test_runner.list_failing_properties() == [EntangledCheckOnUnentangledState]

    def test_entangled_check_on_GHZ_state(self):
        # create an instance of the test runner
        test_runner = TestRunner([EntangledCheckOnGHZState], self.num_inputs, 548, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        # list the failing properties
        assert test_runner.list_passing_properties() == [EntangledCheckOnGHZState]
        assert test_runner.list_failing_properties() == []

    def test_ghz_random_state(self):
        # create an instance of the test runner
        test_runner = TestRunner([EntangledCheckOnRandomState], 1, 547, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        # list the failing properties
        assert test_runner.list_passing_properties() == [EntangledCheckOnRandomState]
        assert test_runner.list_failing_properties() == []
