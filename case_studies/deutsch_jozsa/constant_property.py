from qiskit import QuantumCircuit
from case_studies.deutsch_jozsa.deutsch_jozsa import deutsch_jozsa
from case_studies.deutsch_jozsa.dj_helpers import ConstantOracleInputGenerator
from qucheck.property import Property


class DeutschJozsaWorksForConstantFunction(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        return [ConstantOracleInputGenerator(2, 5)]

    # specify the preconditions for the test
    def preconditions(self, oracle):
        return True

    # specify the operations to be performed on the input
    def operations(self, oracle: QuantumCircuit):
        circ = deutsch_jozsa(oracle)

        # if oracle is constant this should be all 0
        baseline = QuantumCircuit(oracle.num_qubits - 1, oracle.num_qubits - 1)

        self.statistical_analysis.assert_equal(self, list(range(oracle.num_qubits - 1)), circ, list(range(oracle.num_qubits - 1)), baseline, basis=["z"])

