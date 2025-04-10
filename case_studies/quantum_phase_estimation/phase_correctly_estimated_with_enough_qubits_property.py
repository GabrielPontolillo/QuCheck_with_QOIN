# class that inherits from property based test
from qiskit import QuantumCircuit
import random
import numpy as np
import cmath
from fractions import Fraction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from qucheck.property import Property
from qucheck.input_generators import RandomEigenvectorUnitaryPair, RandomUnitaryLimitedDecimals, InputGenerator
from case_studies.quantum_phase_estimation.quantum_phase_estimation import quantum_phase_estimation


class PhaseCorrectlyEstimatedEnoughQubits(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        # here we need to generate inputs i.e. an eigenvector with eigenvalues that are specific fractions of 2pi
        # so if we have enough qubits, we can estimate the phase correctly
        generator_set = FixedEigenvectorUnitaryWithLimitedQubits(1, 1, 1, 3)

        return [generator_set]

    # specify the preconditions for the test
    def preconditions(self, fixed_eigenvector_unitary_with_limited_qubits):
        return True

    # specify the operations to be performed on the input
    def operations(self, fixed_eigenvector_unitary_with_limited_qubits):
        eigenvectors, unitary, estimation_qubits = fixed_eigenvector_unitary_with_limited_qubits

        # perform qpe on with an eigenvector in lower register
        qpe = quantum_phase_estimation(estimation_qubits, UnitaryGate(unitary), eigenvectors[0][0])

        # # state should be the computational basis state corresponding to the phase
        # need to get the phase from eigenvalue, make computational state from eigenvalue
        ph = cmath.phase(eigenvectors[0][1])

        # if problems come up, this should be the first place to look
        # essentially the code below needs to initialise the expected state // last issue was that i needed to reverse the qubits in the statevector
        if ph == 0:
            binary_fraction = "0" * estimation_qubits
        else:
            # ensure phase between 0 and 1
            ph = ph / (2 * np.pi)
            ph = (ph + 1) % 1

            # should be number of estimation qubits
            frac = Fraction(ph).limit_denominator(2 ** estimation_qubits)
            decimal_value = float(frac)

            # Convert fractional part to binary
            binary_fraction = ""

            # limit to number of estimation qubits
            while decimal_value > 0 and len(binary_fraction) <= estimation_qubits:  # limit to prevent infinite loop
                decimal_value *= 2
                bit = int(decimal_value)
                binary_fraction += str(bit)
                decimal_value -= bit

            # we need to pad the end with 0's to make the binary string the correct length
            binary_fraction = binary_fraction + "0" * (estimation_qubits - len(binary_fraction))

        qpe2 = QuantumCircuit(estimation_qubits, estimation_qubits)
        qpe2.initialize(Statevector.from_label(binary_fraction), list(range(estimation_qubits)))

        # need to reverse qubit order to get it to work, probably due to endianness
        self.statistical_analysis.assert_equal(self, list(range(estimation_qubits)), qpe, list(reversed(range(estimation_qubits))), qpe2)


# add input generator for this specific scenario
class FixedEigenvectorUnitaryWithLimitedQubits(InputGenerator):
    # choose the fraction of 2pi to limit the decimals to
    def __init__(self, unitary_qubits_low, unitary_qubits_high, estimation_qubits_low, estimation_qubits_high):
        self.unitary_qubits_low = unitary_qubits_low
        self.unitary_qubits_high = unitary_qubits_high
        self.estimation_qubits_low = estimation_qubits_low
        self.estimation_qubits_high = estimation_qubits_high

    def generate(self, seed):
        random.seed(seed)

        estimation_qubits = random.randint(self.estimation_qubits_low, self.estimation_qubits_high)

        eigenvector_vector_value_pairs, unitary = RandomEigenvectorUnitaryPair(
                RandomUnitaryLimitedDecimals(
                    self.unitary_qubits_low,
                    self.unitary_qubits_high,
                    estimation_qubits), 1).generate(seed)

        return eigenvector_vector_value_pairs, unitary, estimation_qubits

