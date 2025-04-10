# # class that inherits from property based test
# import numpy as np
# from qiskit.circuit.library import UnitaryGate
# from qucheck.property import Property
# from qucheck.input_generators import RandomEigenvectorUnitaryPair, RandomTensorProductOfUnitary, Integer
# from case_studies.quantum_phase_estimation.quantum_phase_estimation import quantum_phase_estimation
#
#
# class PhaseEstimationSumEigenvectors(Property):
#     # specify the inputs that are to be generated
#     def get_input_generators(self):
#         eigenvector_unitary_pair = RandomEigenvectorUnitaryPair(RandomTensorProductOfUnitary(1, 2), 2)
#         estimation_qubits = Integer(1, 3)
#         return [eigenvector_unitary_pair, estimation_qubits]
#
#     # specify the preconditions for the test
#     def preconditions(self, eigenvector_unitary_pair, estimation_qubits):
#         # TODO: this is truly a poor implementation.
#         # to fix, make a pseudorandom unitary generator to pass to RandomEigenvectorUnitaryPair, which allows for a range of eigenvalues
#         # but not infinite, as to allow for multiple eigenvectors with the same eigenvalue
#         # then send all the eigenvectors here instead of an already chosen pair
#
#         # check that the eigenvectors have the same eigenvalue
#         return eigenvector_unitary_pair[0][0][1] == eigenvector_unitary_pair[0][1][1]
#
#     # specify the operations to be performed on the input
#     def operations(self, eigenvector_unitary_pair, estimation_qubits):
#         eigenvectors, unitary = eigenvector_unitary_pair
#
#         # for eigenvector, eigenvalue in eigenvectors:
#         #     print(eigenvalue)
#
#         n = unitary.num_qubits
#
#         # perform qpe on with an eigenvector in lower register
#         qpe = quantum_phase_estimation(estimation_qubits, UnitaryGate(unitary), eigenvectors[0][0])
#
#         # sum of eigenvectors, then normalize
#         normalized_sum_eigenvectors = (eigenvectors[0][0] + eigenvectors[1][0]) / np.sqrt(2)
#         qpe2 = quantum_phase_estimation(estimation_qubits, UnitaryGate(unitary), normalized_sum_eigenvectors)
#
#         # print(qpe)
#         # print(qpe2)
#
#         self.statistical_analysis.assert_equal(self, list(range(estimation_qubits)), qpe,
#                                                list(range(estimation_qubits)), qpe2)
