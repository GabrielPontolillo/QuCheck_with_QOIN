# a test script for the test runner
from unittest import TestCase

from case_studies.quantum_fourier_transform.identity_property import IdentityProperty
from case_studies.quantum_teleportation.input_reg0_equal_to_output_reg2_property import Inq0EqualOutq2
from case_studies.quantum_teleportation.not_teleported_registers_equal_to_plus_property import NotTeleportedPlus
from qucheck.test_runner import TestRunner
from tests.mock_properties.failing_precondition_test_property import FailingPrecondition


class TestTestRunner(TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def tearDown(self):
        TestRunner.property_classes = []
        TestRunner.property_objects = []
        TestRunner.seeds_list_dict = {}

    # test the run_tests method
    def test_run_tests(self):
        # check correct number of circuits generated
        # pass the classes directly
        # create an instance of the test runner
        test_runner = TestRunner([Inq0EqualOutq2], self.num_inputs, 548, self.num_measurements)
        # run the tests
        data = test_runner.run_tests()
        self.assertEqual(test_runner.list_failing_properties(), [])
        self.assertEqual(test_runner.list_passing_properties(), [Inq0EqualOutq2])
        self.assertEqual(data.number_circuits_executed, self.num_inputs * 2 * 3)

    def test_run_tests_same_number_circuits_executed_when_using_same_property_twice(
            self):  # fails, should be automatically fixed when optimisation is fixed
        # create an instance of the test runner
        test_runner = TestRunner([Inq0EqualOutq2, Inq0EqualOutq2], self.num_inputs, 548, self.num_measurements)
        # run the tests
        data = test_runner.run_tests()
        self.assertEqual(test_runner.list_failing_properties(), [])
        self.assertEqual(test_runner.list_passing_properties(), [Inq0EqualOutq2, Inq0EqualOutq2])
        self.assertEqual(data.number_circuits_executed, self.num_inputs * 2 * 3)

        test_runner2 = TestRunner([Inq0EqualOutq2], self.num_inputs, 548, self.num_measurements)
        data = test_runner2.run_tests()
        self.assertEqual(test_runner2.list_failing_properties(), [])
        self.assertEqual(test_runner2.list_passing_properties(), [Inq0EqualOutq2])
        self.assertEqual(data.number_circuits_executed, self.num_inputs * 2 * 3)

    def test_run_tests_different_properties_same_input_optimisation_check(
            self):  # fails, should be automatically fixed when optimisation is fixed
        # create an instance of the test runner
        test_runner = TestRunner([Inq0EqualOutq2, NotTeleportedPlus], self.num_inputs, 548, self.num_measurements)
        # run the tests
        data = test_runner.run_tests()
        self.assertEqual(test_runner.list_failing_properties(), [])
        self.assertEqual(test_runner.list_passing_properties(), [Inq0EqualOutq2, NotTeleportedPlus])
        self.assertEqual(data.number_circuits_executed, (self.num_inputs * 2 * 3) + 3)

        test_runner2 = TestRunner([Inq0EqualOutq2], self.num_inputs, 548, self.num_measurements)
        data = test_runner2.run_tests()
        self.assertEqual(test_runner2.list_failing_properties(), [])
        self.assertEqual(test_runner2.list_passing_properties(), [Inq0EqualOutq2])
        self.assertEqual(data.number_circuits_executed, self.num_inputs * 2 * 3)

        test_runner3 = TestRunner([NotTeleportedPlus], self.num_inputs, 548, self.num_measurements)
        data = test_runner3.run_tests()
        self.assertEqual(test_runner3.list_failing_properties(), [])
        self.assertEqual(test_runner3.list_passing_properties(), [NotTeleportedPlus])
        self.assertEqual(data.number_circuits_executed, (self.num_inputs * 1 * 3) + 3)

    def test_same_seeds(self):
        # create an instance of the test runner
        test_runner = TestRunner([Inq0EqualOutq2, Inq0EqualOutq2], self.num_inputs, 1, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        save_seeds = test_runner.seeds_list_dict
        # create an instance of the test runner
        test_runner2 = TestRunner([Inq0EqualOutq2, Inq0EqualOutq2], self.num_inputs, 1, self.num_measurements)
        # run the tests
        test_runner2.run_tests()
        save_seeds2 = test_runner2.seeds_list_dict
        self.assertEqual(save_seeds, save_seeds2)

    def test_different_seeds(self):
        # create an instance of the test runner
        test_runner = TestRunner([Inq0EqualOutq2, IdentityProperty], self.num_inputs, 1, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        save_seeds = test_runner.seeds_list_dict

        TestRunner.property_classes = []
        TestRunner.property_objects = []
        TestRunner.seeds_list_dict = {}

        test_runner2 = TestRunner([Inq0EqualOutq2, IdentityProperty], self.num_inputs, 2, self.num_measurements)
        # run the tests
        test_runner2.run_tests()
        save_seeds2 = test_runner2.seeds_list_dict
        self.assertNotEqual(save_seeds, save_seeds2)

    def test_failing_precondition(self):
        # create an instance of the test runner
        test_runner = TestRunner([FailingPrecondition], self.num_inputs, 548, self.num_measurements)
        # run the tests
        test_runner.run_tests()
        # list the failing properties
        print("failing properties:")
        print(test_runner.list_failing_properties())
        print("passing properties:")
        # we actually get a list of passing properties objects
        print(test_runner.list_passing_properties())

    def test_repeated_property_does_not_run_more_circuits(self):
        # create an instance of the test runner
        test_runner = TestRunner([Inq0EqualOutq2], self.num_inputs, 1, self.num_measurements)
        # run the tests
        data1 = test_runner.run_tests()
        # create an instance of the test runner
        test_runner2 = TestRunner([Inq0EqualOutq2, Inq0EqualOutq2], self.num_inputs, 1, self.num_measurements)
        data2 = test_runner2.run_tests()
        # run the tests
        self.assertEqual(data1.number_circuits_executed, data2.number_circuits_executed)
