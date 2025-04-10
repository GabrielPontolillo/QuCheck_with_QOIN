import os
import importlib
import inspect
import random
import sys

from qiskit_aer import AerSimulator

from qucheck.property import Property
from qucheck.stats.statistical_analysis_coordinator import TestExecutionStatistics
from qucheck.test_runner import TestRunner


class Coordinator:
    def __init__(self, num_inputs, random_seed=None, alpha=0.01, backend=AerSimulator()):
        self.num_inputs = num_inputs
        self.property_classes = set()
        self.test_runner = None
        self.backend = backend
        self.alpha = alpha

        # this random seed is used to generate other random seeds for the test runner, such that we can replay
        # an entire test run
        if random_seed is None:
            self.random_seed = random.randint(0, 2147483647)
        else:
            self.random_seed = random_seed

    def get_classes(self, folder_path):
        """
        Dynamically import .py files in `folder_path` and discover
        all classes that inherit from `Property`.
        """
        sys.path.insert(0, folder_path)
        for file in os.listdir(folder_path):
            if file.endswith('.py'):
                module = importlib.import_module(file[:-3])
                for name, obj in inspect.getmembers(module):
                    # We only want direct subclasses of `Property` (excluding `Property` itself)
                    if inspect.isclass(obj) and issubclass(obj, Property) and obj is not Property:
                        self.property_classes.add(obj)
        self.property_classes = sorted(self.property_classes, key=lambda x: x.__name__)
        sys.path.pop(0)

    def test(self, path, measurements: int = 2000, run_optimization=True,
             number_of_properties=-1, transpile_gates=None) -> TestExecutionStatistics:
        """
        Run 'IDEAL' or baseline tests on the given set of `Property` classes.
        """
        random.seed(self.random_seed)
        self.get_classes(path)
        print(f"Detected: {self.property_classes}")

        if number_of_properties != -1:
            self.property_classes = set(random.sample(list(self.property_classes), number_of_properties))
        print(f"Used: {self.property_classes}")

        self.test_runner = TestRunner(
            self.property_classes,
            self.num_inputs,
            self.random_seed,
            measurements
        )
        return self.test_runner.run_tests(
            backend=self.backend,
            run_optimization=run_optimization,
            family_wise_p_value=self.alpha,
            transpile_gates=transpile_gates
        )

    def test_QOIN(self, path, measurements: int = 2000, run_optimization=True,
                  number_of_properties=-1, algorithm_name=None, clf=False,
                  transpile_gates=None, noise_model=None, quri=False) -> TestExecutionStatistics:
        random.seed(self.random_seed)
        self.get_classes(path)
        print(f"Detected: {self.property_classes}")

        if number_of_properties != -1:
            self.property_classes = set(random.sample(list(self.property_classes), number_of_properties))
        print(f"Used: {self.property_classes}")

        self.test_runner = TestRunner(
            self.property_classes,
            self.num_inputs,
            self.random_seed,
            measurements
        )
        return self.test_runner.run_tests(
            backend=self.backend,
            run_optimization=run_optimization,
            family_wise_p_value=self.alpha,
            mode="QOIN",
            algorithm_name=algorithm_name,
            clf=clf,
            transpile_gates=transpile_gates,
            noise_model=noise_model,
            quri=quri
        )

    def test_HYBRID(self, path, measurements: int = 2000, run_optimization=True,
                  number_of_properties=-1, algorithm_name=None, clf=False,
                  transpile_gates=None, noise_model=None, quri=False) -> TestExecutionStatistics:
        random.seed(self.random_seed)
        self.get_classes(path)
        print(f"Detected: {self.property_classes}")

        if number_of_properties != -1:
            self.property_classes = set(random.sample(list(self.property_classes), number_of_properties))
        print(f"Used: {self.property_classes}")

        self.test_runner = TestRunner(
            self.property_classes,
            self.num_inputs,
            self.random_seed,
            measurements
        )
        return self.test_runner.run_tests(
            backend=self.backend,
            run_optimization=run_optimization,
            family_wise_p_value=self.alpha,
            mode="HYBRID",
            algorithm_name=algorithm_name,
            clf=clf,
            transpile_gates=transpile_gates,
            noise_model=noise_model,
            quri=quri
        )

    def test_DATA_FARM(self, path, measurements: int = 2000, run_optimization=True,
                  number_of_properties=-1, algorithm_name=None, clf=False,
                  transpile_gates=None, noise_model=None, quri=False, name_mod=0) -> TestExecutionStatistics:
        random.seed(self.random_seed)
        self.get_classes(path)
        print(f"Detected: {self.property_classes}")

        if number_of_properties != -1:
            self.property_classes = set(random.sample(list(self.property_classes), number_of_properties))
        print(f"Used: {self.property_classes}")

        self.test_runner = TestRunner(
            self.property_classes,
            self.num_inputs,
            self.random_seed,
            measurements
        )
        return self.test_runner.run_tests(
            backend=self.backend,
            run_optimization=run_optimization,
            family_wise_p_value=self.alpha,
            mode="DATA_FARM",
            algorithm_name=algorithm_name,
            clf=clf,
            transpile_gates=transpile_gates,
            noise_model=noise_model,
            quri=quri,
            name_mod=name_mod
        )

    def test_NOISE(self, path, measurements: int = 2000, run_optimization=True,
                   number_of_properties=-1, transpile_gates=None, noise_model=None,
                   quri=False) -> TestExecutionStatistics:
        """
        Run tests in 'NOISE' mode, presumably injecting some noise model.
        """
        random.seed(self.random_seed)
        self.get_classes(path)
        print(f"Detected: {self.property_classes}")

        if number_of_properties != -1:
            self.property_classes = set(random.sample(list(self.property_classes), number_of_properties))
        print(f"Used: {self.property_classes}")

        self.test_runner = TestRunner(
            self.property_classes,
            self.num_inputs,
            self.random_seed,
            measurements
        )
        return self.test_runner.run_tests(
            backend=self.backend,
            run_optimization=run_optimization,
            family_wise_p_value=self.alpha,
            mode="NOISE",
            transpile_gates=transpile_gates,
            noise_model=noise_model,
            quri=quri
        )

    def dump_circuits(self, path, run_optimization=True, number_of_properties=-1,
                      name_mod=None, transpile_gates=None) -> TestExecutionStatistics:
        """
        Dumps QASM circuits for the discovered `Property` classes to disk.
        The `name_mod` parameter is appended to filenames to avoid collisions
        when dumping multiple mutants.
        """
        random.seed(self.random_seed)
        self.get_classes(path)
        print(f"Detected: {self.property_classes}")

        if number_of_properties != -1:
            self.property_classes = set(random.sample(list(self.property_classes), number_of_properties))
        print(f"Used: {self.property_classes}")

        # We create a TestRunner with measurements=3000 (arbitrary) for dumping.
        self.test_runner = TestRunner(
            self.property_classes,
            self.num_inputs,
            self.random_seed,
            3000
        )

        return self.test_runner.run_tests(
            backend=self.backend,
            run_optimization=run_optimization,
            family_wise_p_value=self.alpha,
            mode="DUMP",
            name_mod=name_mod,
            transpile_gates=transpile_gates
        )

    def print_outcomes(self):
        if self.test_runner is None:
            raise Exception("No tests have been run yet")

        print("failing properties:")
        failing_properties = self.test_runner.list_failing_properties()

        for prop_obj in self.test_runner.property_objects:
            if type(prop_obj) in failing_properties:
                print("property: ", prop_obj)
                print(self.test_runner.list_inputs(prop_obj))

        print("passing properties:")
        print(self.test_runner.list_passing_properties())