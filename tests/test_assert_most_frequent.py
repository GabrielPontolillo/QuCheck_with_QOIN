# a test script for the test runner
from unittest import TestCase

from qucheck.test_runner import TestRunner
from tests.mock_properties.frequency_test_property import FrequencyProperty, UncertainFrequencyProperty


class TestAssertMostFrequent(TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def tearDown(self):
        TestRunner.property_objects = []
        TestRunner.seeds_list_dict = []

    def test_single_frequency(self):
        # create an instance of the test runner
        test_runner = TestRunner([FrequencyProperty], self.num_inputs, 546, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        # list the failing properties
        assert test_runner.list_passing_properties() == [FrequencyProperty]
        assert test_runner.list_failing_properties() == []

    def test_multiple_frequency(self):
        # create an instance of the test runner
        test_runner = TestRunner([UncertainFrequencyProperty], self.num_inputs, 548, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        # list the failing properties
        assert test_runner.list_passing_properties() == [UncertainFrequencyProperty]
        assert test_runner.list_failing_properties() == []
