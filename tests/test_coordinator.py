# begin testing the coordinator
import os
from unittest import TestCase

from qucheck.coordinator import Coordinator
from qucheck.test_runner import TestRunner

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


class TestCoordinator(TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_measurements = 5000

    def tearDown(self):
        TestRunner.property_classes = []
        TestRunner.property_objects = []
        TestRunner.seeds_list_dict = {}

    # basic test just to see if the coordinator runs all teleportation properties
    def test_coordinator_all_teleportation_properties(self):
        coordinator = Coordinator(self.num_inputs, 1)
        num_circ_executed = coordinator.test(os.path.join(PARENT_DIR, "case_studies/quantum_teleportation"),
                                             self.num_measurements, run_optimization=True).number_circuits_executed

        # test the number of inputs generated
        # all properties should pass
        # test the number of shots taken
        passing = coordinator.test_runner.list_passing_properties()
        print(coordinator.test_runner.list_failing_properties())
        passing = [elem.__name__ for elem in passing]
        self.assertIn("NotTeleportedPlus", passing)
        self.assertIn("Inq0EqualOutq2", passing)
        self.assertIn("UnitaryBeforeAndAfterTeleport", passing)

        failing = coordinator.test_runner.list_failing_properties()
        failing = [elem.__name__ for elem in failing]
        self.assertEqual(len(failing), 0)

        # +3 is the ++ circuits, 6*2 because: (2 circuits per property * 3 basis) * 2 properties that actually generate different circuits
        # self.assertEqual(num_circ_executed, 63)

        # self.assertEqual(coordinator.test_runner.num_measurements, measurements)

    def test_coordinator_all_phase_estimation_properties(self):
        coordinator = Coordinator(self.num_inputs, 106)
        num_circ_executed = coordinator.test(os.path.join(PARENT_DIR, "case_studies/quantum_phase_estimation"),
                                             self.num_measurements, run_optimization=True).number_circuits_executed
        # test the number of inputs generated
        # all properties should pass
        # test the number of shots taken
        passing = coordinator.test_runner.list_passing_properties()
        passing = [elem.__name__ for elem in passing]
        self.assertIn("LowerRegisterUnchangedByEigenvector", passing)
        self.assertIn("PhaseCorrectlyEstimatedEnoughQubits", passing)
        self.assertIn("PhaseEstimationSumDifferentEigenvectors", passing)
        self.assertIn("PhaseEstimationSumEigenvectors", passing)

        failing = coordinator.test_runner.list_failing_properties()
        failing = [elem.__name__ for elem in failing]
        self.assertEqual(len(failing), 0)

        # (2 circuits per property * 3 basis) * 4 properties that actually generate different circuits
        # self.assertEqual(num_circ_executed, 69)

        # self.assertEqual(coordinator.test_runner.num_measurements, measurements)

    def test_coordinator_all_fourier_transform_properties(self):
        coordinator = Coordinator(self.num_inputs, 3)
        num_circ_executed = coordinator.test(os.path.join(PARENT_DIR, "case_studies/quantum_fourier_transform"),
                                             self.num_measurements, run_optimization=True).number_circuits_executed
        # test the number of inputs generated
        # all properties should pass
        # test the number of shots taken
        passing = coordinator.test_runner.list_passing_properties()
        passing = [elem.__name__ for elem in passing]
        self.assertIn("IdentityProperty", passing)
        self.assertIn("LinearShiftToPhaseShift", passing)
        self.assertIn("PhaseShiftToLinearShift", passing)

        failing = coordinator.test_runner.list_failing_properties()
        failing = [elem.__name__ for elem in failing]
        self.assertEqual(len(failing), 0)

        # (2 circuits per property * 3 basis) * 4 properties that actually generate different circuits
        # self.assertEqual(num_circ_executed, 84)

        # self.assertEqual(coordinator.test_runner.num_measurements, measurements)

    def test_coordinator_all_grovers_properties(self):
        coordinator = Coordinator(self.num_inputs, 4)
        num_circ_executed = coordinator.test(os.path.join(PARENT_DIR, "case_studies/grovers_algorithm"),
                                             self.num_measurements, run_optimization=True).number_circuits_executed
        # test the number of inputs generated
        # all properties should pass
        # test the number of shots taken
        passing = coordinator.test_runner.list_passing_properties()
        passing = [elem.__name__ for elem in passing]
        self.assertIn("GroversAlgorithmLowerRegisterMinus", passing)
        self.assertIn("GroversAlgorithmMostFrequentMarked", passing)
        self.assertIn("GroversAlgorithmMostFrequentNotMarkedIfTooManyMarked", passing)

        failing = coordinator.test_runner.list_failing_properties()
        failing = [elem.__name__ for elem in failing]
        self.assertEqual(len(failing), 0)

        # its 24, but need to double check if its correct
        # self.assertEqual(num_circ_executed, 23)
        #
        # self.assertEqual(coordinator.test_runner.num_measurements, measurements)

    def test_coordinator_all_deutsch_jozsa_properties(self):
        coordinator = Coordinator(self.num_inputs, 34)
        num_circ_executed = coordinator.test(os.path.join(PARENT_DIR, "case_studies/deutsch_jozsa"),
                                             self.num_measurements, run_optimization=True).number_circuits_executed
        # test the number of inputs generated
        # all properties should pass
        # test the number of shots taken
        passing = coordinator.test_runner.list_passing_properties()
        passing = [elem.__name__ for elem in passing]
        self.assertIn("DeutschJozsaWorksForConstantFunction", passing)
        self.assertIn("DeutschJozsaWorksForBalancedFunction", passing)
        self.assertIn("DeutschJozsaLowerRegisterMinus", passing)
        self.assertIn("DeutschJozsaVMergeTwoBalancedOracles", passing)
        self.assertIn("DeutschJozsaVMergeTwoConstantOracles", passing)

        failing = coordinator.test_runner.list_failing_properties()
        failing = [elem.__name__ for elem in failing]
        self.assertEqual(len(failing), 0)

        save_seeds = coordinator.test_runner.seeds_list_dict

        print(save_seeds)
        all_seed_list = [seed for seed_list in save_seeds.values() for seed in seed_list]
        print(all_seed_list)

        # depends on the number of constant oracles generated (which shows that it is working) but have to fix value
        # self.assertEqual(num_circ_executed, 45)
        #
        # self.assertEqual(coordinator.test_runner.num_measurements, measurements)

    # test coordinator to check if it will generate the same local seeds with the same random seed
    # also checks if the correct number of inputs are generated if some of the generators are the same
    def test_coordinator_same_seeds_generated_with_same_global_seed(self):
        coordinator = Coordinator(self.num_inputs, 1)
        coordinator.test(os.path.join(PARENT_DIR, "case_studies/quantum_teleportation"), self.num_measurements,
                         run_optimization=True)
        save_seeds = coordinator.test_runner.seeds_list_dict

        # reset the seeds and property objects to ensure next run works
        TestRunner.seeds_list_dict = {}
        TestRunner.property_classes = []
        TestRunner.property_objects = []

        coordinator2 = Coordinator(self.num_inputs, 1)
        coordinator2.test(os.path.join(PARENT_DIR, "case_studies/quantum_teleportation"), self.num_measurements,
                          run_optimization=True)
        save_seeds2 = coordinator2.test_runner.seeds_list_dict

        # add up the sum of the length of all lists in the seed dictionary
        all_seed_list = [seed for seed_list in save_seeds.values() for seed in seed_list]

        # three properties, but two proerties with different input gens
        self.assertEqual(len(all_seed_list), self.num_inputs * 2)

        self.assertDictEqual(save_seeds, save_seeds2)

    # test coordinator to check if it will generate different local seeds with different random seeds
    def test_coordinator_different_seeds_with_different_global_seed(self):
        coordinator = Coordinator(self.num_inputs, 1)
        coordinator.test(os.path.join(PARENT_DIR, "case_studies/quantum_teleportation"), self.num_measurements,
                         run_optimization=True)
        save_seeds = coordinator.test_runner.seeds_list_dict

        # reset the seeds and property objects to ensure next run works
        TestRunner.seeds_list_dict = {}
        TestRunner.property_classes = []
        TestRunner.property_objects = []

        coordinator2 = Coordinator(self.num_inputs, 2)
        coordinator2.test(os.path.join(PARENT_DIR, "case_studies/quantum_teleportation"), self.num_measurements,
                          run_optimization=True)
        save_seeds2 = coordinator2.test_runner.seeds_list_dict

        print(save_seeds.values())
        print(save_seeds2.values())

        all_seed_list = [seed for seed_list in save_seeds.values() for seed in seed_list]
        print(all_seed_list)

        # three properties, but two properties with different input gens
        self.assertEqual(len(all_seed_list), self.num_inputs * 2)
        self.assertNotEqual(save_seeds, save_seeds2)

    def test_coordinator_same_seeds_generated_with_same_global_seed_DJ_version(self):
        coordinator = Coordinator(self.num_inputs, 34)
        coordinator.test(os.path.join(PARENT_DIR, "case_studies/deutsch_jozsa"), self.num_measurements,
                         run_optimization=True)
        save_seeds = coordinator.test_runner.seeds_list_dict
        save_seeds = [seed for seed_list in save_seeds.values() for seed in seed_list]

        # reset the seeds and property objects to ensure next run works
        TestRunner.seeds_list_dict = {}
        TestRunner.property_classes = []
        TestRunner.property_objects = []

        coordinator2 = Coordinator(self.num_inputs, 34)
        coordinator2.test(os.path.join(PARENT_DIR, "case_studies/deutsch_jozsa"), self.num_measurements,
                          run_optimization=True)
        save_seeds2 = coordinator2.test_runner.seeds_list_dict
        save_seeds2 = [seed for seed_list in save_seeds2.values() for seed in seed_list]

        self.assertEqual(save_seeds, save_seeds2)

    # test coordinator to check if it will generate different local seeds with different random seeds
    def test_coordinator_different_seeds_with_different_global_seed_DJ_version(self):
        coordinator = Coordinator(self.num_inputs, 35)
        coordinator.test(os.path.join(PARENT_DIR, "case_studies/deutsch_jozsa"), self.num_measurements,
                         run_optimization=True)
        save_seeds = coordinator.test_runner.seeds_list_dict
        save_seeds = [seed for seed_list in save_seeds.values() for seed in seed_list]

        # reset the seeds and property objects to ensure next run works
        TestRunner.seeds_list_dict = {}
        TestRunner.property_classes = []
        TestRunner.property_objects = []

        coordinator2 = Coordinator(self.num_inputs, 36)
        coordinator2.test(os.path.join(PARENT_DIR, "case_studies/deutsch_jozsa"), self.num_measurements,
                          run_optimization=True)
        save_seeds2 = coordinator2.test_runner.seeds_list_dict
        save_seeds2 = [seed for seed_list in save_seeds2.values() for seed in seed_list]

        self.assertNotEqual(save_seeds, save_seeds2)

    # test coordinator, to check that the property will fail if the preconditions are not met
    def test_coordinator_failing_precondition(self):
        coordinator = Coordinator(self.num_inputs, 902)
        coordinator.test(os.path.join(PARENT_DIR, "tests/mock_properties"), self.num_measurements,
                         run_optimization=True)
        passing = coordinator.test_runner.list_passing_properties()
        passing = [elem.__name__ for elem in passing]
        print(passing)
        self.assertEqual(len(passing), 6)
        self.assertIn("EntangledPrecondition", passing)
        self.assertIn("EntangledCheckOnGHZState", passing)

        failing = coordinator.test_runner.list_failing_properties()
        failing = [elem.__name__ for elem in failing]
        print(failing)
        self.assertEqual(len(failing), 2)
        self.assertIn("FailingPrecondition", failing)
        self.assertIn("EntangledCheckOnUnentangledState", failing)
