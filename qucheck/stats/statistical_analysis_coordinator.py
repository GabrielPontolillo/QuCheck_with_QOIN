import os
import pickle
from time import time
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.providers import Backend
from scipy.stats import fisher_exact

from qucheck.property import Property
from qucheck.stats.assert_entangled import AssertEntangled
from qucheck.utils import HashableQuantumCircuit
from qucheck.stats.assertion import StatisticalAssertion, StandardAssertion, Assertion
from qucheck.stats.measurements import Measurements
from qucheck.stats.single_qubit_distributions.assert_equal import AssertEqual
from qucheck.stats.single_qubit_distributions.assert_different import AssertDifferent
from qucheck.stats.assert_most_frequent import AssertMostFrequent
from qucheck.stats.utils.corrections import holm_bonferroni_correction, holm_bonferroni_correction_second_pass
from qucheck.stats.circuit_generator import CircuitGenerator
import QOIN.Utility as qoin


# this could also be implemented s.t. it samples from the probabilities, but would be sampling twice essentially
# considering the probabilities are already sampled previously
def probability_to_counts(probabilities, shots):
    assert probabilities, "The probabilities are empty!"
    no_round_counts = {k: v * shots for k, v in probabilities.items()}
    rounded_counts = {k: int(round(v)) for k, v in no_round_counts.items()}

    current_sum = sum(rounded_counts.values())
    desired_sum = shots
    difference = desired_sum - current_sum

    if difference != 0:

        fractional_parts = {k: v - int(v) for k, v in no_round_counts.items()}
        adjustment_order = sorted(fractional_parts, key=fractional_parts.get, reverse=True)

        keys = list(adjustment_order)
        total_keys = len(keys)
        for i in range(abs(difference)):
            key = keys[i % total_keys]  # Wrap around if `difference` > number of keys
            rounded_counts[key] += int(np.sign(difference))

    assert sum(rounded_counts.values()) == shots, "Sum of values does not match the expected total!"

    return rounded_counts


class TestExecutionStatistics:
    class FailedProperty:
        def __init__(self, property: Property, failed_classical_assertion: bool, circuits):
            self.property: Property = property
            self.failed_classical_assertion: bool = failed_classical_assertion
            self.circuits = circuits

    def __init__(self) -> None:
        self.number_circuits_executed = 0
        self.failed_property = []


class StatisticalAnalysisCoordinator:
    def __init__(self, number_of_measurements=2000, family_wise_p_value=0.05, mode="TEST", case_study=None) -> None:
        self.assertions_for_property: dict[Property, list[Assertion]] = {}
        self.number_of_measurements = number_of_measurements
        self.family_wise_p_value = family_wise_p_value
        self.mode = mode
        self.case_study = case_study

    #Assertions
    def assert_equal(self, property: Property, qubits1, circuit1: QuantumCircuit, qubits2, circuit2: QuantumCircuit,
                     basis=["x", "y", "z"]):
        # parse qubits so that assert equals always gets sequences of qubits
        if not isinstance(qubits1, Sequence):
            qubits1 = (qubits1,)
        if not isinstance(qubits2, Sequence):
            qubits2 = (qubits2,)
        # hack to make circuits in assert equals be usable as dictionary keys (by ref)
        circ1 = circuit1.copy()
        circ1.__class__ = HashableQuantumCircuit
        circ2 = circuit2.copy()
        circ2.__class__ = HashableQuantumCircuit
        if property in self.assertions_for_property:
            self.assertions_for_property[property].append(AssertEqual(qubits1, circ1, qubits2, circ2, basis))
        else:
            self.assertions_for_property[property] = [AssertEqual(qubits1, circ1, qubits2, circ2, basis)]

    def assert_different(self, property: Property, qubits1, circuit1: QuantumCircuit, qubits2, circuit2: QuantumCircuit,
                         basis=["x", "y", "z"]):
        # parse qubits so that assert equals always gets sequences of qubits
        if not isinstance(qubits1, Sequence):
            qubits1 = (qubits1,)
        if not isinstance(qubits2, Sequence):
            qubits2 = (qubits2,)
        # hack to make circuits in assert equals be usable as dictionary keys (by ref)
        circ1 = circuit1.copy()
        circ1.__class__ = HashableQuantumCircuit
        circ2 = circuit2.copy()
        circ2.__class__ = HashableQuantumCircuit

        if property in self.assertions_for_property:
            self.assertions_for_property[property].append(AssertDifferent(qubits1, circ1, qubits2, circ2, basis))
        else:
            self.assertions_for_property[property] = [AssertDifferent(qubits1, circ1, qubits2, circ2, basis)]

    def assert_entangled(self, property: Property, qubits: Sequence[int], circuit: QuantumCircuit, basis=["z"]):
        # parse qubits so that assert equals always gets sequences of qubits
        if not isinstance(qubits, Sequence):
            qubits = (qubits,)
        # hack to make circuits in assert equals be usable as dictionary keys (by ref)
        circ = circuit.copy()
        circ.__class__ = HashableQuantumCircuit

        if property in self.assertions_for_property:
            self.assertions_for_property[property].append(AssertEntangled(qubits, circ, basis))
        else:
            self.assertions_for_property[property] = [AssertEntangled(qubits, circ, basis)]

    def assert_most_frequent(self, property: Property, qubits, circuit: QuantumCircuit, states, basis=["z"]):
        # parse qubits so that assert equals always gets sequences of qubits / bitstrings
        if not isinstance(qubits, Sequence):
            qubits = (qubits,)
        # hack to make circuits in assert equals be usable as dictionary keys (by ref)
        circ = circuit.copy()
        circ.__class__ = HashableQuantumCircuit

        if property in self.assertions_for_property:
            self.assertions_for_property[property].append(AssertMostFrequent(qubits, circ, states, basis))
        else:
            self.assertions_for_property[property] = [AssertMostFrequent(qubits, circ, states, basis)]

    # Entrypoint for analysis
    def perform_analysis(self, properties: list[Property], backend: Backend, run_optimization: bool, name_mod=0,
                         algorithm_name=None, clf=False, transpile_gates=None, noise_model=None,
                         quri=False) -> TestExecutionStatistics:
        circuit_generator = CircuitGenerator(run_optimization)
        test_execution_stats = TestExecutionStatistics()
        # classical assertion failed dont run quantum
        for property in properties:
            if not property.classical_assertion_outcome:
                test_execution_stats.failed_property.append(
                    TestExecutionStatistics.FailedProperty(property, True, None))
                continue

            for assertion in self.assertions_for_property[property]:
                # Gets all circuits, measurements basis and qubits to measure
                circuit_generator.add_measurement_configuration(assertion.get_measurement_configuration())

        # we pass the circuit generator, which uses the measurement configurations to get the circuits to execute
        # when optimisations are applied, runs the least number of circuits as possible
        # returns measurement object, which is a dictionary of measurements for each basis for each base circuit
        measurements, num_circuits_executed = self._perform_measurements(circuit_generator, backend, name_mod=name_mod,
                                                                         algorithm_name=algorithm_name,
                                                                         clf=clf, transpile_gates=transpile_gates,
                                                                         noise_model=noise_model,
                                                                         quri=quri)

        # this condition is true when dumping circuits
        if measurements is None and num_circuits_executed is None:
            return None

        test_execution_stats.number_circuits_executed = num_circuits_executed
        start_time = time()
        p_values = {}

        # p_values is a dictionary of dictionaries, this will just store them for easier access.
        pv = []

        for property in properties:
            if property.classical_assertion_outcome:
                p_values[property] = {}
                for assertion in self.assertions_for_property[property]:
                    if isinstance(assertion, StatisticalAssertion):
                        p_value = assertion.calculate_p_values(measurements)
                        p_values[property][assertion] = p_value
                        pv.extend(p_value)
                    elif not isinstance(assertion, Assertion):
                        raise ValueError("Assertion must be a subclass of Assertion")

        print("p val calc time", time() - start_time)
        pv = sorted(pv)

        # Only do Holm Bonferroni Correction if there are p_values to correct (preconditions pass)
        if p_values:
            expected_p_values = holm_bonferroni_correction(self.assertions_for_property, p_values,
                                                           self.family_wise_p_value)

        # calculate the outcome of each assertion
        failures_count = 0
        for property in properties:
            if not property.classical_assertion_outcome:
                continue
            for assertion in self.assertions_for_property[property]:
                if isinstance(assertion, StandardAssertion):
                    assertion_outcome = assertion.calculate_outcome(measurements)
                elif isinstance(assertion, StatisticalAssertion):
                    assertion_outcome = assertion.calculate_outcome(p_values[property][assertion],
                                                                    expected_p_values[property][assertion])
                else:
                    raise ValueError("The provided assertions must be a subclass of Assertion")
                if not assertion_outcome:
                    # this is bad, it relies on assertions having circs as attributes so this really is best guess, but since we are passing all measurements
                    # its quite hard to figure out what exactly failed...
                    failed_circuits = []
                    failures_count += 1
                    for _, val in assertion.__dict__.items():
                        if isinstance(val, QuantumCircuit):
                            failed_circuits.append(val)
                    test_execution_stats.failed_property.append(
                        TestExecutionStatistics.FailedProperty(property, False, failed_circuits))
        return test_execution_stats

    # Entrypoint for analysis - new code that tries the hybrid approach, figure out how to call this easily.
    def perform_analysis_hybrid(self, properties: list[Property], backend: Backend, run_optimization: bool, name_mod=0,
                         algorithm_name=None, clf=False, transpile_gates=None, noise_model=None,
                         quri=False) -> TestExecutionStatistics:
        circuit_generator = CircuitGenerator(run_optimization)
        print("performing analysis for hybrid execution")
        test_execution_stats = TestExecutionStatistics()
        # classical assertion failed dont run quantum
        for property in properties:
            if not property.classical_assertion_outcome:
                test_execution_stats.failed_property.append(
                    TestExecutionStatistics.FailedProperty(property, True, None))
                continue

            for assertion in self.assertions_for_property[property]:
                # Gets all circuits, measurements basis and qubits to measure
                circuit_generator.add_measurement_configuration(assertion.get_measurement_configuration())

        # we pass the circuit generator, which uses the measurement configurations to get the circuits to execute
        # when optimisations are applied, runs the least number of circuits as possible
        # returns measurement object, which is a dictionary of measurements for each basis for each base circuit
        self.mode = "NOISE"
        measurements, num_circuits_executed = self._perform_measurements(circuit_generator, backend, name_mod=name_mod,
                                                                         algorithm_name=algorithm_name,
                                                                         clf=clf, transpile_gates=transpile_gates,
                                                                         noise_model=noise_model,
                                                                         quri=quri)

        # this condition is true when dumping circuits
        if measurements is None and num_circuits_executed is None:
            return None

        print(f"initial circuits executed under noise: {num_circuits_executed}")
        test_execution_stats.number_circuits_executed = num_circuits_executed
        p_values = {}

        # p_values is a dictionary of dictionaries, this will just store them for easier access.
        pv = []

        for property in properties:
            if property.classical_assertion_outcome:
                p_values[property] = {}
                for assertion in self.assertions_for_property[property]:
                    if isinstance(assertion, StatisticalAssertion):
                        p_value = assertion.calculate_p_values(measurements)
                        p_values[property][assertion] = p_value
                        pv.extend(p_value)
                    elif not isinstance(assertion, Assertion):
                        raise ValueError("Assertion must be a subclass of Assertion")

        # Only do Holm Bonferroni Correction if there are p_values to correct (preconditions pass)
        if p_values:
            expected_p_values = holm_bonferroni_correction(self.assertions_for_property, p_values,
                                                           self.family_wise_p_value)

        # collect the failing assertions
        initial_failed_assertions = []
        for property in properties:
            if not property.classical_assertion_outcome:
                continue
            for assertion in self.assertions_for_property[property]:
                if isinstance(assertion, StandardAssertion):
                    if not assertion.calculate_outcome(measurements):
                        initial_failed_assertions.append(assertion)
                        assertion.initial_failure = True
                    else:
                        assertion.initial_failure = False
                elif isinstance(assertion, StatisticalAssertion):
                    if not assertion.calculate_outcome(p_values[property][assertion],
                                                       expected_p_values[property][assertion]):
                        initial_failed_assertions.append(assertion)
                        assertion.initial_failure = True
                    else:
                        assertion.initial_failure = False
                else:
                    raise ValueError("The provided assertions must be a subclass of Assertion")

        # generate the subset of circuit that we will need to execute here
        circuit_generator = CircuitGenerator(run_optimization)
        for assertion in initial_failed_assertions:
            circuit_generator.add_measurement_configuration(assertion.get_measurement_configuration())

        self.mode = "QOIN"
        measurements, num_circuits_executed = self._perform_measurements(circuit_generator, backend, name_mod=name_mod,
                                                                         algorithm_name=algorithm_name, clf=clf,
                                                                         transpile_gates=transpile_gates,
                                                                         noise_model=noise_model, quri=quri)

        print(f"circuits executed under qoin: {num_circuits_executed}")

        p_values = {}

        # p_values is a dictionary of dictionaries, this will just store them for easier access.
        pv = []

        for property in properties:
            if property.classical_assertion_outcome:
                p_values[property] = {}
                for assertion in self.assertions_for_property[property]:
                    if isinstance(assertion, StatisticalAssertion) and assertion.initial_failure:
                        p_value = assertion.calculate_p_values(measurements)
                        p_values[property][assertion] = p_value
                        pv.extend(p_value)
                    elif not isinstance(assertion, Assertion):
                        raise ValueError("Assertion must be a subclass of Assertion")

        if p_values:
            expected_p_values = holm_bonferroni_correction_second_pass(self.assertions_for_property, p_values,
                                                                       self.family_wise_p_value)

        # calculate the outcome of each assertion
        failures_count = 0
        for property in properties:
            if not property.classical_assertion_outcome:
                continue
            for assertion in self.assertions_for_property[property]:
                if isinstance(assertion, StandardAssertion):
                    if assertion.initial_failure:
                        assertion_outcome = assertion.calculate_outcome(measurements)
                    else:
                        assertion_outcome = True
                elif isinstance(assertion, StatisticalAssertion):
                    if assertion.initial_failure:
                        assertion_outcome = assertion.calculate_outcome(p_values[property][assertion],
                                                                    expected_p_values[property][assertion])
                    else:
                        assertion_outcome = True
                else:
                    raise ValueError("The provided assertions must be a subclass of Assertion")
                if not assertion_outcome:
                    # this is bad, it relies on assertions having circs as attributes so this really is best guess, but since we are passing all measurements
                    # its quite hard to figure out what exactly failed...
                    failed_circuits = []
                    failures_count += 1
                    for _, val in assertion.__dict__.items():
                        if isinstance(val, QuantumCircuit):
                            failed_circuits.append(val)
                    test_execution_stats.failed_property.append(
                        TestExecutionStatistics.FailedProperty(property, False, failed_circuits))
        return test_execution_stats

    def perform_analysis_data_farm(self, properties: list[Property], backend: Backend, run_optimization: bool,
                                   name_mod=0, algorithm_name=None, clf=False, transpile_gates=None,
                                   noise_model=None, quri=False) -> list[TestExecutionStatistics]:
        """
        Executes all circuits once to gather ideal, noise, and QOIN measurements.
        Then performs four separate analyses:
          1) Single-pass analysis with ideal data
          2) Single-pass analysis with noise data
          3) Single-pass analysis with QOIN data
          4) Hybrid analysis using noise data, then QOIN data for the failing assertions
        Returns them in the order [ideal, noise, qoin, hybrid].
        """
        PATH = os.path.abspath("")
        circuit_generator = CircuitGenerator(run_optimization)

        # Collect measurement configurations
        for property in properties:
            if not property.classical_assertion_outcome:
                # We still skip adding circuits for properties that already failed classical checks
                # but we do not need to record them here (they'll be recorded inside the single-pass/hybrid methods)
                continue
            for assertion in self.assertions_for_property[property]:
                circuit_generator.add_measurement_configuration(assertion.get_measurement_configuration())

        # Execute circuits once, collecting all measurements (ideal, noise, QOIN)
        measurements_id, measurements_noise, measurements_qoin, num_circuits_executed = self._perform_measurements_all(
            circuit_generator,
            backend,
            name_mod=name_mod,
            algorithm_name=algorithm_name,
            clf=clf,
            transpile_gates=transpile_gates,
            noise_model=noise_model,
            quri=quri
        )

        # Save the measurement data and assertion info
        filename_id = f"{PATH}/mutation_test_results/{algorithm_name}/data/ideal_counts_{name_mod}.pkl"
        filename_noise = f"{PATH}/mutation_test_results/{algorithm_name}/data/noise_counts_{name_mod}.pkl"
        filename_qoin = f"{PATH}/mutation_test_results/{algorithm_name}/data/qoin_counts_{name_mod}.pkl"
        filename_assertions = f"{PATH}/mutation_test_results/{algorithm_name}/data/assertions_{name_mod}.pkl"
        filename_circ_gen = f"{PATH}/mutation_test_results/{algorithm_name}/data/circuit_generator_{name_mod}.pkl"
        os.makedirs(os.path.dirname(filename_id), exist_ok=True)

        with open(filename_id, 'wb') as f:
            pickle.dump(measurements_id, f)
        with open(filename_noise, 'wb') as f:
            pickle.dump(measurements_noise, f)
        with open(filename_qoin, 'wb') as f:
            pickle.dump(measurements_qoin, f)
        with open(filename_assertions, 'wb') as f:
            pickle.dump(self.assertions_for_property, f)
        with open(filename_circ_gen, 'wb') as f:
            pickle.dump(circuit_generator, f)

        if measurements_noise is None and num_circuits_executed is None:
            # If we’re dumping circuits only, just return
            return []

        # 1) Single-pass analysis with ideal data
        analysis_ideal = self._analyse_single_pass(properties, measurements_id, num_circuits_executed)

        # 2) Single-pass analysis with noise data
        analysis_noise = self._analyse_single_pass(properties, measurements_noise, num_circuits_executed)

        # 3) Single-pass analysis with QOIN data
        analysis_qoin = self._analyse_single_pass(properties, measurements_qoin, num_circuits_executed)

        # 4) Hybrid analysis: noise pass, then QOIN pass for failing assertions
        analysis_hybrid = self._analyse_hybrid_pass(properties, measurements_noise, measurements_qoin,
                                                    num_circuits_executed)

        return [analysis_ideal, analysis_noise, analysis_qoin, analysis_hybrid]

    def _analyse_single_pass(self, properties: list[Property], measurements: Measurements, num_circuits_executed: int) \
            -> TestExecutionStatistics:
        """
        Performs a single-pass analysis using the given Measurements object.
        Returns a TestExecutionStatistics object for pass/fail results.
        """
        test_execution_stats = TestExecutionStatistics()
        test_execution_stats.number_circuits_executed = num_circuits_executed

        # Collect p-values
        p_values = {}
        collected_pvals = []
        for property in properties:
            if not property.classical_assertion_outcome:
                continue
            p_values[property] = {}
            for assertion in self.assertions_for_property[property]:
                if isinstance(assertion, StatisticalAssertion):
                    p_val = assertion.calculate_p_values(measurements)
                    p_values[property][assertion] = p_val
                    collected_pvals.extend(p_val)
                elif not isinstance(assertion, Assertion):
                    raise ValueError("Assertion must be a subclass of Assertion")

        # Holm-Bonferroni correction
        if p_values:
            expected_p_values = holm_bonferroni_correction(self.assertions_for_property, p_values,
                                                           self.family_wise_p_value)
        else:
            expected_p_values = {}

        # Compute outcomes
        for property in properties:
            if not property.classical_assertion_outcome:
                # Already failed classically
                test_execution_stats.failed_property.append(
                    TestExecutionStatistics.FailedProperty(property, True, None)
                )
                continue

            for assertion in self.assertions_for_property[property]:
                if isinstance(assertion, StandardAssertion):
                    outcome = assertion.calculate_outcome(measurements)
                elif isinstance(assertion, StatisticalAssertion):
                    outcome = assertion.calculate_outcome(
                        p_values[property][assertion],
                        expected_p_values[property][assertion]
                    )
                else:
                    raise ValueError("Assertion must be a subclass of Assertion")

                if not outcome:
                    failed_circuits = []
                    for _, val in assertion.__dict__.items():
                        if isinstance(val, QuantumCircuit):
                            failed_circuits.append(val)
                    test_execution_stats.failed_property.append(
                        TestExecutionStatistics.FailedProperty(property, False, failed_circuits)
                    )

        return test_execution_stats

    def _analyse_hybrid_pass(self, properties: list[Property], measurements_noise: Measurements,
                             measurements_qoin: Measurements, num_circuits_executed: int) -> TestExecutionStatistics:
        """
        Performs a two-pass (hybrid) analysis:
          1) Analyses with measurements_noise, collecting failing assertions
          2) Re-analyses only the failing assertions using measurements_qoin
        Returns a TestExecutionStatistics object reflecting final outcomes.
        """
        test_execution_stats = TestExecutionStatistics()
        test_execution_stats.number_circuits_executed = num_circuits_executed

        # First pass with noise measurements
        p_values_noise = {}
        for property in properties:
            if not property.classical_assertion_outcome:
                continue
            p_values_noise[property] = {}
            for assertion in self.assertions_for_property[property]:
                if isinstance(assertion, StatisticalAssertion):
                    p_val = assertion.calculate_p_values(measurements_noise)
                    p_values_noise[property][assertion] = p_val
                elif not isinstance(assertion, Assertion):
                    raise ValueError("Assertion must be a subclass of Assertion")

        if p_values_noise:
            expected_p_values_noise = holm_bonferroni_correction(
                self.assertions_for_property, p_values_noise, self.family_wise_p_value
            )
        else:
            expected_p_values_noise = {}

        # Determine which assertions fail in the first pass
        failed_assertions = []
        for property in properties:
            if not property.classical_assertion_outcome:
                continue
            for assertion in self.assertions_for_property[property]:
                if isinstance(assertion, StandardAssertion):
                    outcome = assertion.calculate_outcome(measurements_noise)
                else:  # StatisticalAssertion
                    outcome = assertion.calculate_outcome(
                        p_values_noise[property][assertion],
                        expected_p_values_noise[property][assertion]
                    )
                assertion.initial_failure = not outcome
                if not outcome:
                    failed_assertions.append(assertion)

        # Second pass with QOIN for failing assertions
        p_values_qoin = {}
        for property in properties:
            p_values_qoin[property] = {}
            for assertion in failed_assertions:
                if assertion in self.assertions_for_property[property]:
                    if isinstance(assertion, StatisticalAssertion):
                        p_val = assertion.calculate_p_values(measurements_qoin)
                        p_values_qoin[property][assertion] = p_val

        if p_values_qoin:
            expected_p_values_qoin = holm_bonferroni_correction_second_pass(
                self.assertions_for_property, p_values_qoin, self.family_wise_p_value
            )
        else:
            expected_p_values_qoin = {}

        # Compute final outcomes
        for property in properties:
            if not property.classical_assertion_outcome:
                test_execution_stats.failed_property.append(
                    TestExecutionStatistics.FailedProperty(property, True, None)
                )
                continue
            for assertion in self.assertions_for_property[property]:
                # If it failed previously, re-check with QOIN
                if getattr(assertion, 'initial_failure', False):
                    if isinstance(assertion, StandardAssertion):
                        outcome = assertion.calculate_outcome(measurements_qoin)
                    else:  # StatisticalAssertion
                        outcome = assertion.calculate_outcome(
                            p_values_qoin[property][assertion],
                            expected_p_values_qoin[property][assertion]
                        )
                else:
                    # It passed in the noise pass, no need to re-check
                    outcome = True

                if not outcome:
                    failed_circuits = []
                    for _, val in assertion.__dict__.items():
                        if isinstance(val, QuantumCircuit):
                            failed_circuits.append(val)
                    test_execution_stats.failed_property.append(
                        TestExecutionStatistics.FailedProperty(property, False, failed_circuits)
                    )

        return test_execution_stats

    # def perform_analysis_data_farm(self, properties: list[Property], backend: Backend, run_optimization: bool, name_mod=0,
    #                      algorithm_name=None, clf=False, transpile_gates=None, noise_model=None,
    #                      quri=False) -> list[TestExecutionStatistics]:
    #     PATH = os.path.abspath("")
    #     circuit_generator = CircuitGenerator(run_optimization)
    #     print("performing analysis for hybrid execution")
    #     test_execution_stats = TestExecutionStatistics()
    #     # classical assertion failed dont run quantum
    #     for property in properties:
    #         if not property.classical_assertion_outcome:
    #             test_execution_stats.failed_property.append(
    #                 TestExecutionStatistics.FailedProperty(property, True, None))
    #             continue
    #
    #         for assertion in self.assertions_for_property[property]:
    #             # Gets all circuits, measurements basis and qubits to measure
    #             circuit_generator.add_measurement_configuration(assertion.get_measurement_configuration())
    #
    #     # we pass the circuit generator, which uses the measurement configurations to get the circuits to execute
    #     # when optimisations are applied, runs the least number of circuits as possible
    #     # returns measurement object, which is a dictionary of measurements for each basis for each base circuit
    #     measurements_id, measurements_noise, measurements_qoin, num_circuits_executed = self._perform_measurements_all(circuit_generator,
    #                                                                      backend, name_mod=name_mod,
    #                                                                      algorithm_name=algorithm_name,
    #                                                                      clf=clf, transpile_gates=transpile_gates,
    #                                                                      noise_model=noise_model,
    #                                                                      quri=quri)
    #
    #     filename_id = f"{PATH}/mutation_test_results/{algorithm_name}/data/ideal_counts_{name_mod}.pkl"
    #     filename_noise = f"{PATH}/mutation_test_results/{algorithm_name}/data/noise_counts_{name_mod}.pkl"
    #     filename_qoin = f"{PATH}/mutation_test_results/{algorithm_name}/data/qoin_counts_{name_mod}.pkl"
    #     filename_assertions = f"{PATH}/mutation_test_results/{algorithm_name}/data/assertions_{name_mod}.pkl"
    #     filename_circ_gen = f"{PATH}/mutation_test_results/{algorithm_name}/data/circuit_generator_{name_mod}.pkl"
    #     os.makedirs(os.path.dirname(filename_id), exist_ok=True)
    #     with open(filename_id, 'wb') as f:
    #         pickle.dump(measurements_id, f)
    #     with open(filename_noise, 'wb') as f:
    #         pickle.dump(measurements_noise, f)
    #     with open(filename_qoin, 'wb') as f:
    #         pickle.dump(measurements_qoin, f)
    #     with open(filename_assertions, 'wb') as f:
    #         pickle.dump(self.assertions_for_property, f)
    #     with open(filename_circ_gen, 'wb') as f:
    #         pickle.dump(circuit_generator, f)
    #
    #     # this condition is true when dumping circuits
    #     if measurements_noise is None and num_circuits_executed is None:
    #         return None
    #
    #     print(f"initial circuits executed under noise: {num_circuits_executed}")
    #     test_execution_stats.number_circuits_executed = num_circuits_executed
    #     p_values = {}
    #
    #     # p_values is a dictionary of dictionaries, this will just store them for easier access.
    #     pv = []
    #
    #     for property in properties:
    #         if property.classical_assertion_outcome:
    #             p_values[property] = {}
    #             for assertion in self.assertions_for_property[property]:
    #                 if isinstance(assertion, StatisticalAssertion):
    #                     p_value = assertion.calculate_p_values(measurements_noise)
    #                     p_values[property][assertion] = p_value
    #                     pv.extend(p_value)
    #                 elif not isinstance(assertion, Assertion):
    #                     raise ValueError("Assertion must be a subclass of Assertion")
    #
    #     # Only do Holm Bonferroni Correction if there are p_values to correct (preconditions pass)
    #     if p_values:
    #         expected_p_values = holm_bonferroni_correction(self.assertions_for_property, p_values,
    #                                                        self.family_wise_p_value)
    #
    #     # collect the failing assertions
    #     initial_failed_assertions = []
    #     for property in properties:
    #         if not property.classical_assertion_outcome:
    #             continue
    #         for assertion in self.assertions_for_property[property]:
    #             if isinstance(assertion, StandardAssertion):
    #                 if not assertion.calculate_outcome(measurements_noise):
    #                     initial_failed_assertions.append(assertion)
    #                     assertion.initial_failure = True
    #                 else:
    #                     assertion.initial_failure = False
    #             elif isinstance(assertion, StatisticalAssertion):
    #                 if not assertion.calculate_outcome(p_values[property][assertion],
    #                                                    expected_p_values[property][assertion]):
    #                     initial_failed_assertions.append(assertion)
    #                     assertion.initial_failure = True
    #                 else:
    #                     assertion.initial_failure = False
    #             else:
    #                 raise ValueError("The provided assertions must be a subclass of Assertion")
    #
    #     p_values = {}
    #
    #     # p_values is a dictionary of dictionaries, this will just store them for easier access.
    #     pv = []
    #
    #     for property in properties:
    #         if property.classical_assertion_outcome:
    #             p_values[property] = {}
    #             for assertion in self.assertions_for_property[property]:
    #                 if isinstance(assertion, StatisticalAssertion) and assertion.initial_failure:
    #                     p_value = assertion.calculate_p_values(measurements_qoin)
    #                     p_values[property][assertion] = p_value
    #                     pv.extend(p_value)
    #                 elif not isinstance(assertion, Assertion):
    #                     raise ValueError("Assertion must be a subclass of Assertion")
    #
    #     if p_values:
    #         expected_p_values = holm_bonferroni_correction_second_pass(self.assertions_for_property, p_values,
    #                                                                    self.family_wise_p_value)
    #
    #     # calculate the outcome of each assertion
    #     failures_count = 0
    #     for property in properties:
    #         if not property.classical_assertion_outcome:
    #             continue
    #         for assertion in self.assertions_for_property[property]:
    #             if isinstance(assertion, StandardAssertion):
    #                 if assertion.initial_failure:
    #                     assertion_outcome = assertion.calculate_outcome(measurements_qoin)
    #                 else:
    #                     assertion_outcome = True
    #             elif isinstance(assertion, StatisticalAssertion):
    #                 if assertion.initial_failure:
    #                     assertion_outcome = assertion.calculate_outcome(p_values[property][assertion],
    #                                                                     expected_p_values[property][assertion])
    #                 else:
    #                     assertion_outcome = True
    #             else:
    #                 raise ValueError("The provided assertions must be a subclass of Assertion")
    #             if not assertion_outcome:
    #                 # this is bad, it relies on assertions having circs as attributes so this really is best guess, but since we are passing all measurements
    #                 # its quite hard to figure out what exactly failed...
    #                 failed_circuits = []
    #                 failures_count += 1
    #                 for _, val in assertion.__dict__.items():
    #                     if isinstance(val, QuantumCircuit):
    #                         failed_circuits.append(val)
    #                 test_execution_stats.failed_property.append(
    #                     TestExecutionStatistics.FailedProperty(property, False, failed_circuits))
    #     return test_execution_stats

    def _perform_measurements(self, circuit_generator: CircuitGenerator, backend: Backend, name_mod=0,
                              algorithm_name=None,
                              clf=False, transpile_gates=None, noise_model=None, quri=False) -> (Measurements, int):
        if self.case_study is None:
            "CHANGE THE CASE STUDY NAME"
        start_time = time()
        measurements = Measurements()
        print("before get circuits")
        circuits_to_execute = circuit_generator.get_circuits_to_execute()
        print("optim time", time() - start_time)
        if len(circuits_to_execute) == 0:
            return measurements, 0
        # equivalent mutants would get transpiled to the same circuit, so we should run with no optimisation for mutation testing
        start_time = time()
        print("num circuits to transpile", len(circuits_to_execute))
        # optimisation level 0 is no optimisation - this is necessary for the mutation testing with equivalent mutants
        # maybe an issue here, should perhaps create the correct backend
        transpiled_circuits = transpile(circuits_to_execute, backend, optimization_level=0,
                                        basis_gates=transpile_gates, )
        print("num circuits transpiled", len(transpiled_circuits))
        # print("len transpiled circs", sum([len(circ.data) for circ in transpiled_circuits])/len(transpiled_circuits))
        print("transpilation time", time() - start_time)

        print(f"BEGIN MODE {self.mode}")

        if self.mode == "DUMP":
            PATH = os.path.abspath("")
            filename = f"{PATH}/dumps/circuits_{name_mod}.pkl"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(transpiled_circuits, f)
            return None, None
        elif self.mode == "TEST":
            start_time = time()
            results = backend.run(transpiled_circuits, shots=self.number_of_measurements).result().get_counts()
            print("circuit execution time", time() - start_time)
            if len(circuits_to_execute) == 1:
                results = (results,)
            for counts, circuit, transpiled_circuit in zip(results, circuits_to_execute, transpiled_circuits):
                for measurement_name, original_circuit in circuit_generator.get_measurement_info(circuit):
                    measurements.add_measurement(original_circuit, measurement_name, counts)
            return measurements, len(transpiled_circuits)
        elif self.mode == "NOISE":
            start_time = time()
            if quri:
                results = qoin.execute_circuit_quri(qc=transpiled_circuits, shots=self.number_of_measurements,
                                                    ideal_run=False, noisy_run=True)
            else:
                results = qoin.execute_circuit(qc=transpiled_circuits, shots=self.number_of_measurements,
                                               ideal_run=False, noisy_run=True, noise_model=noise_model)
            print("circuit execution time", time() - start_time)
            for counts, circuit, transpiled_circuit in zip(results, circuits_to_execute, transpiled_circuits):
                for measurement_name, original_circuit in circuit_generator.get_measurement_info(circuit):
                    measurements.add_measurement(original_circuit, measurement_name, counts)
            return measurements, len(transpiled_circuits)
        elif self.mode == "QOIN":
            start_time = time()

            if clf:
                model = qoin.Qoin_load_clf(algorithm_name)
                print("clf true")
            else:
                model = qoin.Qoin_load(algorithm_name, "Z")

            results = []
            for circuit in transpiled_circuits:
                # probabilities = qoin.filtered_result_clf_quri(model, circuit, self.number_of_measurements)
                if quri:
                    if clf:
                        probabilities = qoin.filtered_result_clf_quri(model, circuit, self.number_of_measurements)
                    else:
                        probabilities = qoin.filtered_result_quri(model, circuit, self.number_of_measurements)
                else:
                    if clf:
                        probabilities = qoin.filtered_result_clf(model, circuit, noise_model)
                    else:
                        probabilities = qoin.filtered_result(model, circuit, noise_model)

                print(probabilities)
                assert probabilities, "Probabilities is empty!"
                for key in probabilities:
                    assert 0 <= probabilities[key] <= 1, f"Probability out of range: {probabilities[key]}"

                rounded_counts = probability_to_counts(probabilities, self.number_of_measurements)
                print(rounded_counts)
                results.append(rounded_counts)

            print("circuit execution time", time() - start_time)
            for counts, circuit, transpiled_circuit in zip(results, circuits_to_execute, transpiled_circuits):
                for measurement_name, original_circuit in circuit_generator.get_measurement_info(circuit):
                    measurements.add_measurement(original_circuit, measurement_name, counts)
            return measurements, len(transpiled_circuits)

    # creates a dictionary of measurements for each assertion,
    def _perform_measurements_all(self, circuit_generator: CircuitGenerator, backend: Backend, name_mod=0, algorithm_name=None,
                              clf=False, transpile_gates=None, noise_model=None, quri=False) -> tuple[Measurements, Measurements, Measurements, int]:
        if self.case_study is None:
            "CHANGE THE CASE STUDY NAME"
        start_time = time()
        measurements_ideal = Measurements()
        measurements_noise = Measurements()
        measurements_qoin = Measurements()
        print("before get circuits")
        circuits_to_execute = circuit_generator.get_circuits_to_execute()
        print("optim time", time()-start_time)
        if len(circuits_to_execute) == 0:
            return measurements_ideal, measurements_noise, measurements_qoin, 0
        # equivalent mutants would get transpiled to the same circuit, so we should run with no optimisation for mutation testing
        start_time = time()
        print("num circuits to transpile", len(circuits_to_execute))
        # optimisation level 0 is no optimisation - this is necessary for the mutation testing with equivalent mutants
        # maybe an issue here, should perhaps create the correct backend
        transpiled_circuits = transpile(circuits_to_execute, backend, optimization_level=0, basis_gates=transpile_gates,)
        print("num circuits transpiled", len(transpiled_circuits))
        # print("len transpiled circs", sum([len(circ.data) for circ in transpiled_circuits])/len(transpiled_circuits))
        print("transpilation time", time()-start_time)

        print(f"BEGIN MODE {self.mode}")

        model = qoin.Qoin_load_clf(algorithm_name)
        results_qoin = []
        for circuit in transpiled_circuits:
            probabilities = qoin.filtered_result_clf(model, circuit, noise_model)
            probabilities = probability_to_counts(probabilities, self.number_of_measurements)
            results_qoin.append(probabilities)
        results_noise = qoin.execute_circuit(qc=transpiled_circuits, shots=self.number_of_measurements, ideal_run=False,
                                       noisy_run=True, noise_model=noise_model)
        results_id = backend.run(transpiled_circuits, shots=self.number_of_measurements).result().get_counts()

        noise_pvals = []
        qoin_pvals = []

        for idx in range(len(results_qoin)):
            results_qoin[idx] = {key: results_qoin[idx][key] for key in sorted(results_qoin[idx])}
            results_noise[idx] = {key: results_noise[idx][key] for key in sorted(results_noise[idx])}
            results_id[idx] = {key: results_id[idx][key] for key in sorted(results_id[idx])}

            # print("\n==========  Execution result ===========")
            # print("pre transpiled circuit")
            # print(circuits_to_execute[idx].draw(fold=300, cregbundle=True))
            # print(f"QOIN: {results_qoin[idx]}")
            # print(f"Noise: {results_noise[idx]}")
            # print(f"Ideal: {results_id[idx]}")

            # Compute counts for each dictionary
            # qoin_counts = compute_index_counts(results_qoin[idx])
            # noise_counts = compute_index_counts(results_noise[idx])
            # id_counts = compute_index_counts(results_id[idx])

            # print("\n==========  Individual qubit filtering ===========")
            # print(f"QOIN: {qoin_counts}")
            # print(f"Noise: {noise_counts}")
            # print(f"Ideal: {id_counts}")
            # print("========================================")

            # Perform Fisher's Exact Test for Dict1 vs Dict2 and Dict1 vs Dict3
            # id_vs_noise = fisher_test_index_counts(id_counts, noise_counts)
            # id_vs_qoin = fisher_test_index_counts(id_counts, qoin_counts)
            #
            # print("Fisher's Exact Test Results id vs noise:")
            # for index, result in id_vs_noise.items():
            #     noise_pvals.append(result['p_value'])
            #     print(result['p_value'])
            #
            # print("\nFisher's Exact Test Results id vs qoin:")
            # for index, result in id_vs_qoin.items():
            #     qoin_pvals.append(result['p_value'])
            #     print(result['p_value'])

        # print("\n========================================")
        # for i in [1e-2, 1e-6, 1e-12, 1e-24]:
        #     print(
        #         f"Number of p-values less than {i} for noise: {count_the_number_of_values_less_than_a_threshold(noise_pvals, i)}")
        # for i in [1e-2, 1e-6, 1e-12, 1e-24]:
        #     print(
        #         f"Number of p-values less than {i} for QOIN: {count_the_number_of_values_less_than_a_threshold(qoin_pvals, i)}")
        # print("========================================")
        # print(f"total number of p-values {len(noise_pvals)}")
        # print(
        #     f"number of failing p-values for noise {len(holm_bonferroni_correction2(noise_pvals, family_wise_alpha=0.01))}")
        # print(f"total number of p-values {len(qoin_pvals)}")
        # print(
        #     f"number of failing p-values for QOIN {len(holm_bonferroni_correction2(qoin_pvals, family_wise_alpha=0.01))}")
        # print("========================================")


        print("circuit execution time", time()-start_time)
        for qoin_counts, noise_counts, ideal_counts, circuit, transpiled_circuit in zip(results_qoin, results_noise, results_id,
                                                                                        circuits_to_execute, transpiled_circuits):
            # maps circuits to original circuits, find the measurement basis for the original circuit
            for measurement_name, original_circuit in circuit_generator.get_measurement_info(circuit):
                measurements_ideal.add_measurement(original_circuit, measurement_name, ideal_counts)
                measurements_noise.add_measurement(original_circuit, measurement_name, noise_counts)
                measurements_qoin. add_measurement(original_circuit, measurement_name, qoin_counts)
        return measurements_ideal, measurements_noise, measurements_qoin, len(transpiled_circuits)


def count_the_number_of_values_less_than_a_threshold(p_values, threshold):
    count = 0
    for p_value in p_values:
        if p_value < threshold:
            count += 1
    return count


def holm_bonferroni_correction2(p_vals, family_wise_alpha=0.05):
    ret_list = []
    p_vals = sorted(p_vals)
    for i, p_val in enumerate(p_vals):
        # if last one, print it
        if i == len(p_vals) - 1:
            print(f"new alpha {family_wise_alpha / (len(p_vals) - i)}")
            print(f"last pval {p_val}")
        if i == 0:
            print(f"new alpha {family_wise_alpha / (len(p_vals) - i)}")
            print(f"first pval {p_val}")
        if p_val <= (family_wise_alpha / (len(p_vals) - i)):
            ret_list.append(p_val)
    return ret_list


# Function to compute counts of 0s and 1s for each index
def compute_index_counts(dictionary):
    max_index = max(len(key) for key in dictionary.keys())
    index_counts = {i: {'0': 0, '1': 0} for i in range(max_index)}

    for key, count in dictionary.items():
        for i, bit in enumerate(key):
            index_counts[i][bit] += count
    return index_counts


# Function to compute Fisher's Exact Test for index counts
def fisher_test_index_counts(counts1, counts2):
    results = {}

    for index in counts1.keys():
        table = [
            [counts1[index]['0'], counts2[index]['0']],
            [counts1[index]['1'], counts2[index]['1']]
        ]
        odds_ratio, p_value = fisher_exact(table)
        results[index] = {'odds_ratio': odds_ratio, 'p_value': p_value}

    return results