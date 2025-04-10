# from qiskit import QuantumCircuit
# from case_studies.deutsch_jozsa.deutsch_jozsa import deutsch_jozsa
# from case_studies.deutsch_jozsa.dj_helpers import ConstantOracleInputGenerator, vmerge
# from qucheck.property import Property
#
#
# class DeutschJozsaVMergeTwoConstantOracles(Property):
#     # specify the inputs that are to be generated
#     def get_input_generators(self):
#         return [ConstantOracleInputGenerator(2, 5), ConstantOracleInputGenerator(2, 5)]
#
#     # specify the preconditions for the test
#     def preconditions(self, oracle1, oracle2):
#         return True
#
#     # specify the operations to be performed on the input
#     def operations(self, oracle1: QuantumCircuit, oracle2: QuantumCircuit):
#         circ = deutsch_jozsa(vmerge(oracle1, oracle2))
#
#         # if oracle is constant this should be all 0
#         baseline = QuantumCircuit(circ.num_qubits - 1, circ.num_qubits - 1)
#
#         self.statistical_analysis.assert_equal(self, list(range(circ.num_qubits - 1)), circ, list(range(oracle1.num_qubits - 1)), baseline, basis=["z"])