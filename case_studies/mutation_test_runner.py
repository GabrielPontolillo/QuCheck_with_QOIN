import csv
import gc
import importlib.util
import math
import os
import pickle
import sys
import time
import numpy as np
import concurrent.futures
from unittest.mock import patch

import pandas as pd
from nbformat.sign import algorithms
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError
from scipy.stats import fisher_exact

from QOIN import Utility as qoin
from qucheck.coordinator import Coordinator
from qucheck.stats.assertion import StatisticalAssertion, StandardAssertion
from qucheck.stats.single_qubit_distributions.assert_different import AssertDifferent
from qucheck.stats.single_qubit_distributions.assert_equal import AssertEqual
from qucheck.test_runner import TestRunner
from qucheck.stats.circuit_generator import CircuitGenerator

PATH = os.path.abspath("")


def hellinger_distance(p, q):
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)

    # Calculate the squared difference of square roots
    diff_sq = np.sum((sqrt_p - sqrt_q) ** 2)

    # Return the Hellinger distance
    return np.sqrt(diff_sq / 2)


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


def compute_index_counts(dictionary):
    max_index = max(len(key) for key in dictionary.keys())
    index_counts = {i: {'0': 0, '1': 0} for i in range(max_index)}

    for key, count in dictionary.items():
        for i, bit in enumerate(key):
            index_counts[i][bit] += count
    return index_counts


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


def count_the_number_of_values_less_than_a_threshold(p_values, threshold):
    count = 0
    for p_value in p_values:
        if p_value < threshold:
            count += 1
    return count


def delete_files_in_dumps():
    dumps_dir = os.path.join(os.getcwd(), "dumps")

    if not os.path.exists(dumps_dir):
        raise FileNotFoundError(f"The subdirectory 'dumps' does not exist in the current directory.")

    if not os.path.isdir(dumps_dir):
        raise NotADirectoryError(f"The path {dumps_dir} is not a directory.")

    print(f"Deleting dump files")
    for filename in os.listdir(dumps_dir):
        if filename.endswith(".qasm") or filename.endswith(".pkl"):
            file_path = os.path.join(dumps_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


def ibm_fez_noise_model():
    noise_model = NoiseModel()

    # Define T1 and T2 relaxation (convert microseconds to seconds)
    T1 = 137.55e-6  # T1 mean
    T2 = 82.61e-6  # T2 mean
    gate_time = 68e-9  # Gate time in seconds (from caption)

    # Add thermal relaxation error for single-qubit gates
    thermal_error = thermal_relaxation_error(T1, T2, gate_time)
    noise_model.add_all_qubit_quantum_error(thermal_error, ['id', 'sx', 'x'])

    # Add depolarizing errors for single-qubit gates
    ID_error = 0.000443
    depolarizing_error_single = depolarizing_error(ID_error, 1)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_single, ['id', 'sx', 'x'])

    # RZ gate does not have errors

    # Add thermal relaxation error for CZ gate
    thermal_error_cz = thermal_relaxation_error(T1, T2, gate_time * 10).tensor(
        thermal_relaxation_error(T1, T2, gate_time)
    )
    noise_model.add_all_qubit_quantum_error(thermal_error_cz, ['cz'])

    # Add depolarizing error for CZ gate
    CZ_error = 0.002848
    depolarizing_error_cz = depolarizing_error(CZ_error, 2)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_cz, ['cz'])

    # Add readout error
    prob_measure_0_prepare_1 = 0.029385
    prob_measure_1_prepare_0 = 0.019849
    readout_error_matrix = [
        [1 - prob_measure_1_prepare_0, prob_measure_1_prepare_0],  # P(measure 0 | prepare 0)
        [prob_measure_0_prepare_1, 1 - prob_measure_0_prepare_1]  # P(measure 1 | prepare 1)
    ]
    readout_error = ReadoutError(readout_error_matrix)
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def equiv_noise_model():
    noise_model = NoiseModel()

    error = 0.02
    depolarizing_error_single = depolarizing_error(error, 1)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_single, ['id', 'sx', 'x', 'rz'])

    two_error = 0.1
    depolarizing_error_two = depolarizing_error(two_error, 2)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_two, ['cx'])

    return noise_model


def import_function(module_str, path, function_name):
    spec = importlib.util.spec_from_file_location(module_str, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_str] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def cleanup_test_runner():
    TestRunner.property_classes = []
    TestRunner.property_objects = []
    TestRunner.seeds_list_dict = {}
    TestRunner.num_inputs = 0
    TestRunner.do_shrinking = None
    TestRunner.max_attempts = 0
    TestRunner.num_measurements = 0
    TestRunner.test_execution_stats = None
    gc.collect()  # Force garbage collection


def test_and_store(algorithm_name, optimisation, mode="IDEAL", clf=False, transpile_gates=None, noise_model=None,
                   quri=False):
    inputs = [200]
    shots = [3000]
    number_of_properties_list = [3]

    for input_val in inputs:
        for measurements in shots:
            for number_of_properties in number_of_properties_list:
                print(
                    f"number of inputs: {input_val}, number of measurements: {measurements}, number of properties: {number_of_properties}")
                filename = f"mutation_test_results/{algorithm_name}/{algorithm_name}_{input_val}_{measurements}_{number_of_properties}_mt_results.csv"
                dir_path = os.path.dirname(filename)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                with open(filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([
                        "Mutant Name",
                        "Number of Properties",
                        "Number of Inputs",
                        "Number of Measurements",
                        "Mode",
                        "Result",
                        "Number of Circuits Executed",
                        "Number of Unique Failed Properties",
                        "Number of Failed Properties",
                        "Unique Failed Properties",
                        "Time Taken"
                    ])

                    # for i in range(1):
                    #     run_single_test(
                    #         algorithm_name, input_val, measurements, "em", i,
                    #         number_of_properties, run_optimization=optimisation,
                    #         csvwriter=csvwriter, mode=mode, clf=clf,
                    #         transpile_gates=transpile_gates, noise_model=noise_model,
                    #         quri=quri
                    #     )

                    # Regular mutants
                    for i in range(10):
                        run_single_test(
                            algorithm_name, input_val, measurements, "m", i,
                            number_of_properties, run_optimization=optimisation,
                            csvwriter=csvwriter, mode=mode, clf=clf,
                            transpile_gates=transpile_gates, noise_model=noise_model,
                            quri=quri
                        )
                    #
                    # Equivalent mutants
                    for i in range(5):
                        run_single_test(
                            algorithm_name, input_val, measurements, "em", i,
                            number_of_properties, run_optimization=optimisation,
                            csvwriter=csvwriter, mode=mode, clf=clf,
                            transpile_gates=transpile_gates, noise_model=noise_model,
                            quri=quri
                        )


def run_single_test(algorithm_name, num_inputs, measurements, mutant_type, index, number_of_properties,
                    run_optimization=True, csvwriter=None,
                    mode="IDEAL", clf=False, transpile_gates=None,
                    noise_model=None, quri=False):
    mutant_name = f"{algorithm_name}_{mutant_type}{index}"

    circuit_function = import_function(
        mutant_name,
        f"{PATH}/{algorithm_name}/mutants/{mutant_name}.py",
        algorithm_name
    )
    print(f"Testing {mutant_name}")

    with patch(f"case_studies.{algorithm_name}.{algorithm_name}.{algorithm_name}", circuit_function):
        reload_classes(f"{PATH}/{algorithm_name}")
        backend = AerSimulator(method='statevector')
        backend.set_options(
            max_parallel_threads=0,
            max_parallel_experiments=0,
            max_parallel_shots=1,
            statevector_parallel_threshold=8
        )

        coordinator = Coordinator(num_inputs, backend=backend)

        start = time.time()
        if mode == "NOISE":
            result = coordinator.test_NOISE(
                f"{PATH}/{algorithm_name}",
                measurements,
                run_optimization=run_optimization,
                number_of_properties=number_of_properties,
                transpile_gates=transpile_gates,
                noise_model=noise_model,
                quri=quri
            )
        elif mode == "QOIN":
            result = coordinator.test_QOIN(
                f"{PATH}/{algorithm_name}",
                measurements,
                run_optimization=run_optimization,
                number_of_properties=number_of_properties,
                algorithm_name=algorithm_name,
                clf=clf,
                transpile_gates=transpile_gates,
                noise_model=noise_model,
                quri=quri
            )
        elif mode == "HYBRID":
            result = coordinator.test_HYBRID(
                f"{PATH}/{algorithm_name}",
                measurements,
                run_optimization=run_optimization,
                number_of_properties=number_of_properties,
                algorithm_name=algorithm_name,
                clf=clf,
                transpile_gates=transpile_gates,
                noise_model=noise_model,
                quri=quri
            )
        elif mode == "DATA_FARM":
            mod_label = f"{mutant_type}{index}"  # e.g. "m0", "em4", etc.
            results_list = coordinator.test_DATA_FARM(
                f"{PATH}/{algorithm_name}",
                measurements,
                run_optimization=run_optimization,
                number_of_properties=number_of_properties,
                algorithm_name=algorithm_name,
                clf=clf,
                transpile_gates=transpile_gates,
                noise_model=noise_model,
                quri=quri,
                name_mod=mod_label
            )
            end = time.time()
            # results_list is [analysis_ideal, analysis_noise, analysis_qoin, analysis_hybrid]
            pass_names = ["IDEAL", "NOISE", "QOIN", "HYBRID"]

            # For each of the four results, create a separate CSV line
            for test_execution_stats, pass_name in zip(results_list, pass_names):
                # If you want a single "outcome" (Fail/Pass), mark "Fail" if there were any failed properties
                failures = test_execution_stats.failed_property
                outcome = "Fail" if failures else "Pass"

                # Count how many (and which) properties failed
                property_count = {}
                for failed_prop in failures:
                    prop_class_name = failed_prop.property.__class__.__name__
                    property_count[prop_class_name] = property_count.get(prop_class_name, 0) + 1

                failed_property_string = " & ".join(f"{p}: {count}" for p, count in property_count.items())
                num_unique_failed_properties = len(property_count)
                num_failed_properties = len(failures)

                # Build the row for this pass
                row = [
                    mutant_name,  # e.g. quantum_fourier_transform_m0
                    number_of_properties,  # e.g. 3
                    num_inputs,  # e.g. 50
                    measurements,  # e.g. 3000
                    pass_name,  # e.g. IDEAL
                    outcome,  # "Fail" or "Pass"
                    str(test_execution_stats.number_circuits_executed),
                    str(num_unique_failed_properties),
                    str(num_failed_properties),
                    failed_property_string,
                    str(end - start)  # total time, same as before
                ]
                if csvwriter:
                    csvwriter.writerow(row)
            # no need to do more
            return None
        elif mode == "DUMP":
            # Pass "m0", "m1", etc. or "em0", "em1", etc. as the unique suffix
            mod_label = f"{mutant_type}{index}"  # e.g. "m0", "em2", etc.

            result = coordinator.dump_circuits(
                f"{PATH}/{algorithm_name}",
                run_optimization=run_optimization,
                number_of_properties=number_of_properties,
                transpile_gates=transpile_gates,
                name_mod=mod_label
            )
            return None
        else:  # "IDEAL"
            result = coordinator.test(
                f"{PATH}/{algorithm_name}",
                measurements,
                run_optimization=run_optimization,
                number_of_properties=number_of_properties,
                transpile_gates=transpile_gates
            )
        end = time.time()

        num_circuits_executed = result.number_circuits_executed

        # Gather and count each failed property type
        failed_properties = result.failed_property
        property_count = {}
        for property_ in failed_properties:
            prop_class_name = property_.property.__class__.__name__
            property_count[prop_class_name] = property_count.get(prop_class_name, 0) + 1

        # Build the failed_property_string with counts, e.g. "PropA: 3 & PropB: 5"
        failed_property_string = ""
        for prop_name, count in property_count.items():
            failed_property_string += f"{prop_name}: {count} & "
        if failed_property_string.endswith(" & "):
            failed_property_string = failed_property_string[:-3]

        # The total number of *unique* failed properties and total failures
        num_unique_failed_properties = len(property_count)
        num_failed_properties = len(failed_properties)

        outcome = "Fail" if num_failed_properties > 0 else "Pass"

        result_row = [
            mutant_name,
            number_of_properties,
            num_inputs,
            measurements,
            mode,
            outcome,
            str(num_circuits_executed),
            str(num_unique_failed_properties),
            str(num_failed_properties),
            failed_property_string,
            str(end - start)
        ]

        if csvwriter:
            csvwriter.writerow(result_row)

        print(f"Finished testing {mutant_name}")
        return result_row


def reload_classes(folder_path):
    sys.path.insert(0, folder_path)
    for file in os.listdir(folder_path):
        if file.endswith('.py'):
            module = importlib.import_module(file[:-3])
            importlib.reload(module)
    sys.path.pop(0)


def merge_csv_files(algorithm_name, name_mod=None):
    directory = f"mutation_test_results/{algorithm_name}/"
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                 f.endswith('mt_results.csv') and not f.endswith("_merged_results.csv")]

    dataframes = []
    for file in all_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        if name_mod:
            merged_filename = os.path.join(directory, f"{algorithm_name}_{name_mod}_merged_results.csv")
        else:
            merged_filename = os.path.join(directory, f"{algorithm_name}_merged_results.csv")
        merged_df.to_csv(merged_filename, index=False)
        print(f"Merged results saved to {merged_filename}")

        # Delete individual CSV files
        for file in all_files:
            os.remove(file)
            print(f"Deleted: {file}")
    else:
        print(f"No CSV files found in {directory}")


def train_model(algorithm_name, optimisation, num_inputs_to_train_with, mode="IDEAL", clf=False,
                transpile_gates=None, noise_model=None, quri=False):
    coordinator = Coordinator(num_inputs_to_train_with)

    # coordinator.dump_circuits(
    #     f"{PATH}/{algorithm_name}",
    #     run_optimization=optimisation,
    #     transpile_gates=transpile_gates,
    #     name_mod="original"
    # )

    for i in range(10):
        run_single_test(
            algorithm_name,
            num_inputs=num_inputs_to_train_with,
            measurements=3000,  # measurements dont matter, wont be executed
            mutant_type="m",
            index=i,
            number_of_properties=-1,
            run_optimization=optimisation,
            mode="DUMP",
            transpile_gates=transpile_gates,
            noise_model=noise_model,
            quri=quri,
        )

    for i in range(5):
        run_single_test(
            algorithm_name,
            num_inputs=num_inputs_to_train_with,
            measurements=3000,
            mutant_type="em",
            index=i,
            number_of_properties=-1,
            run_optimization=optimisation,
            mode="DUMP",
            transpile_gates=transpile_gates,
            noise_model=noise_model,
            quri=quri,
        )

    if quri:
        qoin.QOIN_data_generation_file_quri(
            path="./dumps",
            case_study_name=algorithm_name,
        )
    else:
        qoin.QOIN_data_generation_file(
            path="./dumps",
            case_study_name=algorithm_name,
            noise_model=noise_model
        )

    # 7) Train the classifier or the regular Qoin model
    # if clf:
    qoin.Qoin_train_clf(case_study_name=algorithm_name, basis="Z")
    # else:
    qoin.Qoin_train(case_study_name=algorithm_name, basis="Z")

    # 8) Finally, remove all .qasm dumps
    delete_files_in_dumps()


def execute_conditions(algorithm_name, optimisation, transpile_gates, noise_model=None, quri=False, clf=False,
                       modes=("IDEAL", "NOISE", "QOIN", "HYBRID"), num_inputs_to_train_with=50, train=True):
    """
    Executes the test and store, merge CSV, and train model functions for specified conditions.

    Args:
        algorithm_name (str): Name of the algorithm being executed.
        optimisation (bool): Whether optimisation is enabled.
        transpile_gates (list): Gates to be used for transpilation.
        noise_model (object, optional): Noise model to be used for NOISE and QOIN conditions. Defaults to None.
        quri (bool): Whether to use Quri simulation.
        clf (bool): Whether classification is enabled.
        modes (tuple): Conditions to be executed (e.g., "IDEAL", "NOISE", "QOIN"). Defaults to all.
    """
    if train:
        print("- Begin Training Model -")
        train_model(algorithm_name, optimisation, num_inputs_to_train_with,
                    mode="QOIN",
                    clf=clf,
                    transpile_gates=transpile_gates,
                    noise_model=noise_model,
                    quri=quri)

    if "IDEAL" in modes:
        test_and_store(algorithm_name, optimisation, mode="IDEAL", transpile_gates=transpile_gates)
        merge_csv_files(algorithm_name, name_mod="IDEAL")

    if "NOISE" in modes:
        test_and_store(algorithm_name, optimisation, mode="NOISE", transpile_gates=transpile_gates,
                       noise_model=noise_model, quri=quri)
        merge_csv_files(algorithm_name, name_mod=f"NOISE_{transpile_gates.__str__()}_quri{quri}")

    if "QOIN" in modes:
        test_and_store(algorithm_name, optimisation,
                       mode="QOIN",
                       clf=clf,
                       transpile_gates=transpile_gates,
                       noise_model=noise_model,
                       quri=quri)
        merge_csv_files(algorithm_name, name_mod=f"QOIN_{transpile_gates.__str__()}_quri{quri}_clf{clf}")

    if "HYBRID" in modes:
        test_and_store(algorithm_name, optimisation,
                       mode="HYBRID",
                       clf=clf,
                       transpile_gates=transpile_gates,
                       noise_model=noise_model,
                       quri=quri)
        merge_csv_files(algorithm_name, name_mod=f"HYBRID_{transpile_gates.__str__()}_quri{quri}_clf{clf}")

    if "DATA_FARM" in modes:
        test_and_store(algorithm_name, optimisation,
                       mode="DATA_FARM",
                       clf=clf,
                       transpile_gates=transpile_gates,
                       noise_model=noise_model,
                       quri=quri)
        merge_csv_files(algorithm_name, name_mod=f"ALL_{transpile_gates.__str__()}_quri{quri}_clf{clf}")

    print("- Finished -")


def gather_p_values_and_distances(algorithm_name, mutant_label):
    """
    Internal helper that performs the p-value comparisons for a specific mutant label.
    E.g. 'em0', 'em1', 'm0', 'm9', etc.
    This is the renamed version of the old compare_p_values function,
    except that we replace hard-coded 'em0' references with the passed-in mutant_label.
    """
    import sys
    import_path = f'{os.path.dirname(PATH)}/case_studies/{algorithm_name}/'
    print(f"Importing from {import_path}")
    sys.path.append(import_path)

    # Using the mutant_label to load the correct pickled files
    with open(f"{PATH}/mutation_test_results/{algorithm_name}/data/ideal_counts_{mutant_label}.pkl", 'rb') as f:
        results_id = pickle.load(f)
    with open(f"{PATH}/mutation_test_results/{algorithm_name}/data/noise_counts_{mutant_label}.pkl", 'rb') as f:
        results_noise = pickle.load(f)
    with open(f"{PATH}/mutation_test_results/{algorithm_name}/data/qoin_counts_{mutant_label}.pkl", 'rb') as f:
        results_qoin = pickle.load(f)
    with open(f"{PATH}/mutation_test_results/{algorithm_name}/data/assertions_{mutant_label}.pkl", 'rb') as f:
        assertions = pickle.load(f)

    qoin_pvals = []
    noise_pvals = []
    id_pvals = []
    hellinger_qoin_distances = []
    hellinger_noise_distances = []
    hellinger_eq_id_distances = []
    hellinger_eq_noise_distances = []
    hellinger_eq_qoin_distances = []

    # Collect p-values
    for property in assertions.keys():
        id_temp = {
            "property_name": property.__class__.__name__,
            "p_values": []
        }
        qoin_temp = {
            "property_name": property.__class__.__name__,
            "p_values": []
        }
        qoin_temp2 = {
            "property_name": property.__class__.__name__,
            "distances": []
        }
        noise_temp = {
            "property_name": property.__class__.__name__,
            "p_values": []
        }
        noise_temp2 = {
            "property_name": property.__class__.__name__,
            "distances": []
        }
        id_dist_assert_eq_or_diff = {
            "property_name": property.__class__.__name__,
            "distances": []
        }
        noise_dist_assert_eq_or_diff = {
            "property_name": property.__class__.__name__,
            "distances": []
        }
        qoin_dist_assert_eq_or_diff = {
            "property_name": property.__class__.__name__,
            "distances": []
        }
        for assertion in assertions[property]:
            if isinstance(assertion, StatisticalAssertion):
                qoin_temp["p_values"].extend(assertion.calculate_p_values(results_qoin))
                noise_temp["p_values"].extend(assertion.calculate_p_values(results_noise))
                id_temp["p_values"].extend(assertion.calculate_p_values(results_id))
            elif isinstance(assertion, StandardAssertion):
                qoin_temp["p_values"].extend([assertion.calculate_outcome(results_qoin)])
                noise_temp["p_values"].extend([assertion.calculate_outcome(results_noise)])
                id_temp["p_values"].extend([assertion.calculate_outcome(results_id)])

            # returns list of dicts with all circuit measurements
            counts_id = assertion.get_measurements_from_circuits(results_id)
            counts_noise = assertion.get_measurements_from_circuits(results_noise)
            counts_qoin = assertion.get_measurements_from_circuits(results_qoin)

            if isinstance(assertion, StatisticalAssertion):
                # compare circuits 2 by two if assert equal or different
                if isinstance(assertion, AssertEqual) or isinstance(assertion, AssertDifferent):
                    for i in range(0, len(counts_id), 2):
                        all_bitstrings_id = set(counts_id[i].keys()).union(counts_id[i+1].keys())
                        all_bitstrings_noise = set(counts_noise[i].keys()).union(counts_noise[i+1].keys())
                        all_bitstrings_qoin = set(counts_qoin[i].keys()).union(counts_qoin[i+1].keys())

                        total_id = sum(counts_id[i].values())
                        total_noise = sum(counts_noise[i].values())
                        total_qoin = sum(counts_qoin[i].values())

                        prob_id_0 = np.array([counts_id[i].get(bitstring, 0) / total_id for bitstring in all_bitstrings_id])
                        prob_id_1 = np.array([counts_id[i+1].get(bitstring, 0) / total_id for bitstring in all_bitstrings_id])

                        prob_noise_0 = np.array([counts_noise[i].get(bitstring, 0) / total_noise for bitstring in all_bitstrings_noise])
                        prob_noise_1 = np.array([counts_noise[i+1].get(bitstring, 0) / total_noise for bitstring in all_bitstrings_noise])

                        prob_qoin_0 = np.array([counts_qoin[i].get(bitstring, 0) / total_qoin for bitstring in all_bitstrings_qoin])
                        prob_qoin_1 = np.array([counts_qoin[i+1].get(bitstring, 0) / total_qoin for bitstring in all_bitstrings_qoin])

                        hellinger_id = hellinger_distance(prob_id_0, prob_id_1)
                        hellinger_noise = hellinger_distance(prob_noise_0, prob_noise_1)
                        hellinger_qoin = hellinger_distance(prob_qoin_0, prob_qoin_1)

                        id_dist_assert_eq_or_diff["distances"].append(hellinger_id)
                        noise_dist_assert_eq_or_diff["distances"].append(hellinger_noise)
                        qoin_dist_assert_eq_or_diff["distances"].append(hellinger_qoin)

            # for each circuit measurement dict, we need to calculate the hellinger distance
            for i in range(len(counts_id)):
                all_bitstrings = set(counts_id[i].keys()).union(counts_noise[i].keys(), counts_qoin[i].keys())

                # Create aligned probability distributions
                total_id = sum(counts_id[i].values())
                total_noise = sum(counts_noise[i].values())
                total_qoin = sum(counts_qoin[i].values())

                prob_id = np.array([counts_id[i].get(bitstring, 0) / total_id for bitstring in all_bitstrings])
                prob_noise = np.array([counts_noise[i].get(bitstring, 0) / total_noise for bitstring in all_bitstrings])
                prob_qoin = np.array([counts_qoin[i].get(bitstring, 0) / total_qoin for bitstring in all_bitstrings])

                # Calculate Hellinger distances
                hellinger_noise = hellinger_distance(prob_id, prob_noise)
                hellinger_qoin = hellinger_distance(prob_id, prob_qoin)

                qoin_temp2["distances"].append(hellinger_qoin)
                noise_temp2["distances"].append(hellinger_noise)



        qoin_pvals.append(qoin_temp)
        noise_pvals.append(noise_temp)
        id_pvals.append(id_temp)
        hellinger_qoin_distances.append(qoin_temp2)
        hellinger_noise_distances.append(noise_temp2)
        hellinger_eq_id_distances.append(id_dist_assert_eq_or_diff)
        hellinger_eq_noise_distances.append(noise_dist_assert_eq_or_diff)
        hellinger_eq_qoin_distances.append(qoin_dist_assert_eq_or_diff)

    return hellinger_noise_distances, noise_pvals, hellinger_qoin_distances, qoin_pvals, id_pvals, hellinger_eq_id_distances, hellinger_eq_noise_distances, hellinger_eq_qoin_distances


def gather_all_p_values_and_distances_to_csv(algorithm_name):
    """
    Gathers Hellinger distances and p-values for all mutants (em and m) and writes the results to a CSV file.

    Args:
        algorithm_name (str): Name of the algorithm for which to gather data.
        output_csv (str): Path to the output CSV file (default: "p_values_and_distances.csv").

    Returns:
        pd.DataFrame: The DataFrame containing all gathered data.
    """
    # List to collect rows for the DataFrame
    data_rows = []

    # Define ranges for em and m mutants
    em_range = range(5)  # em0 through em4
    m_range = range(10)  # m0 through m9

    # Helper function to process each mutant
    def process_mutant(mutant_type, index_range):
        for i in index_range:
            mutant_label = f"{mutant_type}{i}"
            print(f"Processing mutant: {mutant_label}")
            try:
                hellinger_noise, noise_pvals, hellinger_qoin, qoin_pvals, id_pvals, hellinger_eq_id, hellinger_eq_noise, hellinger_eq_qoin = gather_p_values_and_distances(
                    algorithm_name, mutant_label
                )

                # Add a row for the mutant
                data_rows.append({
                    "mutant_label": mutant_label,
                    "mutant_type": mutant_type,
                    "hellinger_noise_distances": hellinger_noise,
                    "hellinger_qoin_distances": hellinger_qoin,
                    "noise_pvals": noise_pvals,
                    "qoin_pvals": qoin_pvals,
                    "id_pvals": id_pvals,
                    "hellinger_id_eq_distances": hellinger_eq_id,
                    "hellinger_noise_eq_distances": hellinger_eq_noise,
                    "hellinger_qoin_eq_distances": hellinger_eq_qoin
                })
            except Exception as e:
                print(f"Error processing mutant {mutant_label}: {e}")

    # Process em and m mutants
    process_mutant("em", em_range)
    process_mutant("m", m_range)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data_rows)

    directory = f"mutation_test_results/{algorithm_name}/"

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(directory, f"{algorithm_name}_qoin_closeness_vs_p_val.csv"), index=False)

    return df


def get_width_and_depth(algorithm_name, mutant_label):
    import sys

    max_width = 0
    min_width = math.inf
    max_depth = 0
    min_depth = math.inf

    import_path = f'{os.path.dirname(PATH)}/case_studies/{algorithm_name}/'
    sys.path.append(import_path)

    # Using the mutant_label to load the correct pickled files
    with open(f"{PATH}/mutation_test_results/{algorithm_name}/data/assertions_{mutant_label}.pkl", 'rb') as f:
        assertions = pickle.load(f)

    for property in assertions.keys():
        for assertion in assertions[property]:
            # Process assertions with two circuits (e.g. AssertEqual or AssertDifferent)
            if isinstance(assertion, StatisticalAssertion):
                if isinstance(assertion, AssertEqual) or isinstance(assertion, AssertDifferent):
                    # for circuit in [assertion.circuit1, assertion.circuit2]:
                    for circuit in [assertion.circuit1]:
                        # print(circuit)
                        cdepth = circuit.depth()
                        cwidth = circuit.num_qubits
                        if cwidth > max_width:
                            max_width = cwidth
                        if cwidth < min_width:
                            min_width = cwidth
                        if cdepth > max_depth:
                            max_depth = cdepth
                        if cdepth < min_depth:
                            min_depth = cdepth
            else:
                # Process single-circuit assertions
                # print(assertion.circuit)
                cdepth = assertion.circuit.depth()
                cwidth = assertion.circuit.num_qubits
                if cwidth > max_width:
                    max_width = cwidth
                if cwidth < min_width:
                    min_width = cwidth
                if cdepth > max_depth:
                    max_depth = cdepth
                if cdepth < min_depth:
                    min_depth = cdepth

    return max_width, min_width, max_depth, min_depth


def get_width_and_depth_transpiled(algorithm_name, mutant_label):
    import sys

    max_width = 0
    min_width = math.inf
    max_depth = 0
    min_depth = math.inf

    import_path = f'{os.path.dirname(PATH)}/case_studies/{algorithm_name}/'
    sys.path.append(import_path)

    # Using the mutant_label to load the correct pickled files
    with open(f"{PATH}/mutation_test_results/{algorithm_name}/data/assertions_{mutant_label}.pkl", 'rb') as f:
        assertions = pickle.load(f)

    for property in assertions.keys():
        for assertion in assertions[property]:
            # Process assertions with two circuits (e.g. AssertEqual or AssertDifferent)
            if isinstance(assertion, StatisticalAssertion):
                if isinstance(assertion, AssertEqual) or isinstance(assertion, AssertDifferent):
                    # for circuit in [assertion.circuit1, assertion.circuit2]:
                    for circuit in [assertion.circuit1]:
                        # print(circuit)
                        transpiled_circuit = transpile(circuit, AerSimulator(method='statevector'),
                                                       optimization_level=0,
                                                       basis_gates=['id', 'sx', 'x', 'cz', 'rz'])
                        cdepth = transpiled_circuit.depth()
                        cwidth = transpiled_circuit.num_qubits
                        if cwidth > max_width:
                            max_width = cwidth
                        if cwidth < min_width:
                            min_width = cwidth
                        if cdepth > max_depth:
                            max_depth = cdepth
                        if cdepth < min_depth:
                            min_depth = cdepth
            else:
                # Process single-circuit assertions
                circuit = assertion.circuit
                transpiled_circuit = transpile(circuit, AerSimulator(method='statevector'),
                                               optimization_level=0,
                                               basis_gates=['id', 'sx', 'x', 'cz', 'rz'])
                cdepth = transpiled_circuit.depth()
                cwidth = transpiled_circuit.num_qubits
                if cwidth > max_width:
                    max_width = cwidth
                if cwidth < min_width:
                    min_width = cwidth
                if cdepth > max_depth:
                    max_depth = cdepth
                if cdepth < min_depth:
                    min_depth = cdepth

    return max_width, min_width, max_depth, min_depth


def gather_all_width_and_depth(algorithm_name):
    """
    Gathers minimum and maximum circuit depths and widths for all mutants (em and m)
    and prints the aggregated results.

    Args:
        algorithm_name (str): Name of the algorithm for which to gather data.

    Returns:
        dict: A dictionary containing the overall max/min width and max/min depth.
    """
    # List to collect rows for the DataFrame if needed later
    data_rows = []

    # Define ranges for em and m mutants
    em_range = range(5)  # em0 through em4
    m_range = range(10)  # m0 through m9

    print(f"Processing algorithm: {algorithm_name}")

    # Helper function to process each mutant type
    def process_mutant(mutant_type, index_range):
        max_width = 0
        min_width = math.inf
        max_depth = 0
        min_depth = math.inf

        for i in index_range:
            mutant_label = f"{mutant_type}{i}"
            print(f"Processing mutant: {mutant_label}")
            w, mw, d, md = get_width_and_depth(algorithm_name, mutant_label)
            if w > max_width:
                max_width = w
            if mw < min_width:
                min_width = mw
            if d > max_depth:
                max_depth = d
            if md < min_depth:
                min_depth = md
        return max_width, min_width, max_depth, min_depth

    def process_mutant_transpiled(mutant_type, index_range):
        max_width = 0
        min_width = math.inf
        max_depth = 0
        min_depth = math.inf

        for i in index_range:
            mutant_label = f"{mutant_type}{i}"
            print(f"Processing mutant: {mutant_label}")
            w, mw, d, md = get_width_and_depth_transpiled(algorithm_name, mutant_label)
            if w > max_width:
                max_width = w
            if mw < min_width:
                min_width = mw
            if d > max_depth:
                max_depth = d
            if md < min_depth:
                min_depth = md
        return max_width, min_width, max_depth, min_depth

    # Process both mutant types
    w1, mw1, d1, md1 = process_mutant("em", em_range)
    w2, mw2, d2, md2 = process_mutant("m", m_range)

    overall_max_width = max(w1, w2)
    overall_min_width = min(mw1, mw2)
    overall_max_depth = max(d1, d2)
    overall_min_depth = min(md1, md2)

    print(f"Overall maximum width: {overall_max_width}")
    print(f"Overall minimum width: {overall_min_width}")
    print(f"Overall maximum depth: {overall_max_depth}")
    print(f"Overall minimum depth: {overall_min_depth}")

    # Process both mutant types
    w1, mw1, d1, md1 = process_mutant_transpiled("em", em_range)
    w2, mw2, d2, md2 = process_mutant_transpiled("m", m_range)

    overall_max_width_t = max(w1, w2)
    overall_min_width_t = min(mw1, mw2)
    overall_max_depth_t = max(d1, d2)
    overall_min_depth_t = min(md1, md2)

    print(f"Overall maximum width transpiled: {overall_max_width_t}")
    print(f"Overall minimum width transpiled: {overall_min_width_t}")
    print(f"Overall maximum depth transpiled: {overall_max_depth_t}")
    print(f"Overall minimum depth transpiled: {overall_min_depth_t}")

    return {
        "max_width": overall_max_width,
        "min_width": overall_min_width,
        "max_depth": overall_max_depth,
        "min_depth": overall_min_depth,
        "max_width transpiled": overall_max_width_t,
        "min_width transpiled": overall_min_width_t,
        "max_depth transpiled": overall_max_depth_t,
        "min_depth transpiled": overall_min_depth_t
    }




if __name__ == "__main__":
    for algorithm in ["quantum_teleportation"]:
        execute_conditions(
            algorithm_name=algorithm,
            optimisation=True,
            transpile_gates=['id', 'sx', 'x', 'cz', 'rz'],
            noise_model=ibm_fez_noise_model(),
            quri=False,
            clf=True,
            modes=("QOIN",),
            num_inputs_to_train_with=20,
            # if train true will try to dump, which causes error bc we have new perform meas
            # set to false when doing hybrid mode
            train=False
    )
    #
    # algorithms = [
    #     "grovers_algorithm",
    #     "deutsch_jozsa",
    #     "quantum_teleportation",
    #     "quantum_phase_estimation",
    #     "quantum_fourier_transform"
    # ]
    #
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(gather_all_p_values_and_distances_to_csv, algorithms)
    #
    # gather_all_width_and_depth("grovers_algorithm")
    # gather_all_width_and_depth("deutsch_jozsa")
    # gather_all_width_and_depth("quantum_teleportation")
    # gather_all_width_and_depth("quantum_phase_estimation")
    # gather_all_width_and_depth("quantum_fourier_transform")

