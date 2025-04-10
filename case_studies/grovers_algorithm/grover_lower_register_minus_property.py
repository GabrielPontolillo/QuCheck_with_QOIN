import numpy as np

from qiskit import QuantumCircuit
from case_studies.grovers_algorithm.grovers_algorithm import grovers_algorithm
from case_studies.grovers_algorithm.grovers_algorithm_helpers import RandomGroversOracleMarkedStatesPairGenerator
from qucheck.property import Property


class GroversAlgorithmLowerRegisterMinus(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        return [RandomGroversOracleMarkedStatesPairGenerator(3, 5)]

    # specify the preconditions for the test
    def preconditions(self, oracle_pair):
        oracle, marked_states = oracle_pair
        if len(marked_states) == 0 or len(marked_states) > 2**(oracle.num_qubits - 1) - 1:
            return False
        return True

    # specify the operations to be performed on the input
    def operations(self, oracle_pair):
        oracle, marked_states = oracle_pair

        # one qubit is workspace
        N = 2**(oracle.num_qubits-1)
        # number of marked states is used to identify the number of grover iterations to apply
        M = len(marked_states)

        # src Nielsen and Chuang, quantum computation and quantum information
        n_iterations = int(np.floor((np.pi/4) * np.sqrt((N/M))))

        circ = grovers_algorithm(oracle, n_iterations)

        # should be -
        baseline = QuantumCircuit(1, 1)
        baseline.x(0)
        baseline.h(0)

        self.statistical_analysis.assert_equal(self, [circ.num_qubits - 1], circ, [0], baseline)

