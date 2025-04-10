# a test script for the test runner
import copy
import os
import pickle
from unittest import TestCase

from qiskit import qasm2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.stats import fisher_exact
from tqdm import tqdm

from case_studies.mutation_test_runner import ibm_fez_noise_model
from case_studies.quantum_fourier_transform.identity_property import IdentityProperty
from case_studies.quantum_teleportation.input_reg0_equal_to_output_reg2_property import Inq0EqualOutq2
from case_studies.quantum_teleportation.not_teleported_registers_equal_to_plus_property import NotTeleportedPlus
from qucheck.test_runner import TestRunner
from tests.mock_properties.failing_precondition_test_property import FailingPrecondition
from QOIN import Utility as qoin
from qucheck.stats.statistical_analysis_coordinator import probability_to_counts


def equiv_noise_model():
    noise_model = NoiseModel()

    error = 0.02
    depolarizing_error_single = depolarizing_error(error, 1)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_single, ['id', 'sx', 'x', 'rz'])

    two_error = 0.1
    depolarizing_error_two = depolarizing_error(two_error, 2)
    noise_model.add_all_qubit_quantum_error(depolarizing_error_two, ['cx'])

    return noise_model

# returns a list of failing pvals
def holm_bonferroni_correction(p_vals, family_wise_alpha=0.05):
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

def count_the_number_of_values_less_than_a_threshold(p_values, threshold):
    count = 0
    for p_value in p_values:
        if p_value < threshold:
            count += 1
    return count

def fishers_test(dict1, dict2):
    p_vals = []

    # Convert the first key to a string and get its length
    key_length = len(str(list(dict1.keys())[0]))  # Convert key to string

    for i in range(key_length):  # Iterate over bit positions
        contingency_table = [[0, 0], [0, 0]]

        # Process the first dictionary
        for bitstring, count in dict1.items():
            bitstring = str(bitstring)  # Ensure bitstring is treated as a string
            if bitstring[-(i + 1)] == "0":  # Access the character from the right
                contingency_table[0][0] += count
            else:
                contingency_table[0][1] += count

        # Process the second dictionary
        for bitstring, count in dict2.items():
            bitstring = str(bitstring)  # Ensure bitstring is treated as a string
            if bitstring[-(i + 1)] == "0":  # Access the character from the right
                contingency_table[1][0] += count
            else:
                contingency_table[1][1] += count

        # Perform Fisher's Exact Test
        _, p_value = fisher_exact(contingency_table)
        p_vals.append(p_value)

    return p_vals


class TestTestRunner(TestCase):
    def tearDown(self):
        TestRunner.property_classes = []
        TestRunner.property_objects = []
        TestRunner.seeds_list_dict = {}
        TestRunner.num_inputs = 0
        TestRunner.do_shrinking = None
        TestRunner.max_attempts = 0
        TestRunner.num_measurements = 0
        TestRunner.test_execution_stats = None


    # test the run_tests method
    def test_run_tests(self):
        # qiskit import from dumps
        path = os.path.dirname(os.path.abspath(""))
        path = path + "/test_qoin/dumps_test"

        files = os.listdir(path)

        All_input_list = [path + "/" + x for x in files]

        noise = ibm_fez_noise_model()

        case_study_name = "quantum_teleportation"
        p2 = os.path.dirname(os.path.dirname(os.path.abspath(""))) + f"/QOIN/models/clf{case_study_name}.models"
        with open(f"{p2}", 'rb') as f:
            model = pickle.load(f)

        noise_pvals = []
        qoin_pvals = []

        for single_input in tqdm(All_input_list):
            qc = qasm2.load(single_input, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

            execution_result = qoin.execute_circuit([copy.deepcopy(qc) for _ in range(1)], ideal_run=True, noisy_run=True,
                                                    noise_model=noise)
            print("\n==========  Execution result ===========")
            ideal = {key: execution_result["ideal"][key] for key in sorted(execution_result["ideal"])}
            noisy = {key: execution_result["noisy"][key] for key in sorted(execution_result["noisy"])}

            probabilities = qoin.filtered_result_clf(model, copy.deepcopy(qc), noise_model=noise)

            probabilities = probability_to_counts(probabilities, 1024)
            assert sum(probabilities.values()) == 1024

            probabilities = {key: probabilities[key] for key in sorted(probabilities)}

            print(ideal)
            print(noisy)
            print(probabilities)

            # Compute counts for each dictionary
            dict1_counts = compute_index_counts(ideal)
            dict2_counts = compute_index_counts(noisy)
            dict3_counts = compute_index_counts(probabilities)

            print("\n==========  Individual qubit filtering ===========")
            print(f"Ideal: {dict1_counts}")
            print(f"Noise: {dict2_counts}")
            print(f"Qoin: {dict3_counts}")
            print("========================================")

            # Perform Fisher's Exact Test for Dict1 vs Dict2 and Dict1 vs Dict3
            dict1_vs_dict2 = fisher_test_index_counts(dict1_counts, dict2_counts)
            dict1_vs_dict3 = fisher_test_index_counts(dict1_counts, dict3_counts)

            # dict1_vs_dict2b = fishers_test(ideal, noisy)
            # dict1_vs_dict3b = fishers_test(ideal, probabilities)

            # Output results
            print("Fisher's Exact Test Results (Dict1 vs Dict2):")
            for index, result in dict1_vs_dict2.items():
                noise_pvals.append(result['p_value'])
                print(result['p_value'])

            print("\nFisher's Exact Test Results (Dict1 vs Dict3):")
            for index, result in dict1_vs_dict3.items():
                qoin_pvals.append(result['p_value'])
                print(result['p_value'])

        print("\n========================================")
        for i in [1e-2, 1e-6, 1e-12, 1e-24]:
            print(f"Number of p-values less than {i} for noise: {count_the_number_of_values_less_than_a_threshold(noise_pvals, i)}")
        for i in [1e-2, 1e-6, 1e-12, 1e-24]:
            print(f"Number of p-values less than {i} for QOIN: {count_the_number_of_values_less_than_a_threshold(qoin_pvals, i)}")
        print("========================================")
        print(f"total number of p-values {len(noise_pvals)}")
        print(f"number of failing p-values for noise {len(holm_bonferroni_correction(noise_pvals, family_wise_alpha=0.01))}")
        print(f"total number of p-values {len(qoin_pvals)}")
        print(f"number of failing p-values for QOIN {len(holm_bonferroni_correction(qoin_pvals, family_wise_alpha=0.01))}")
        print("========================================")

    def test2(self):
        a = AerSimulator()
        b = AerSimulator(noise_model=ibm_fez_noise_model())
        print("guard")
