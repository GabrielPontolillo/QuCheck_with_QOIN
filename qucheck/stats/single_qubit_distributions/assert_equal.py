from typing import Sequence
from uuid import uuid4
from scipy import stats as sci

from qucheck.utils import HashableQuantumCircuit
from qucheck.stats.assertion import StatisticalAssertion
from qucheck.stats.measurement_configuration import MeasurementConfiguration
from qucheck.stats.measurements import Measurements
from qucheck.stats.utils.common_measurements import measure_x, measure_y, measure_z


class AssertEqual(StatisticalAssertion):
    def __init__(self, qubits1: Sequence[int], circuit1: HashableQuantumCircuit, qubits2: Sequence[int], circuit2: HashableQuantumCircuit, basis = ["x", "y", "z"]) -> None:
    # TODO: add a clause for lists of qubits instead of single registers
        super().__init__()
        self.qubits1 = qubits1
        self.circuit1 = circuit1
        self.qubits2 = qubits2
        self.circuit2 = circuit2
        self.basis = basis
        self.measurement_ids = {basis: uuid4() for basis in basis}

    def calculate_p_values(self, measurements: Measurements) -> list[float]:
        p_vals = []
        for qubit1, qubit2 in zip(self.qubits1, self.qubits2):
            for basis in self.basis:
                qubit1_counts = measurements.get_counts(self.circuit1, self.measurement_ids[basis])
                qubit2_counts = measurements.get_counts(self.circuit2, self.measurement_ids[basis])
                if not isinstance(qubit1_counts, dict) or not isinstance(qubit2_counts, dict):
                    if not set(map(type, qubit1_counts.values())) == {int} or not set(map(type, qubit2_counts.values())) == {int}:
                        raise ValueError("The measurements are not in the correct format")
                contingency_table = [[0, 0], [0, 0]]
                for bitstring, count in qubit1_counts.items():
                    if bitstring[len(bitstring) - qubit1 - 1] == "0":
                        contingency_table[0][0] += count
                    else:
                        contingency_table[0][1] += count
                for bitstring, count in qubit2_counts.items():
                    if bitstring[len(bitstring) - qubit2 - 1] == "0":
                        contingency_table[1][0] += count
                    else:
                        contingency_table[1][1] += count
                _, p_value = sci.fisher_exact(contingency_table)
                p_vals.append(p_value)
        return p_vals

    def calculate_outcome(self, p_values: Sequence[float], expected_p_values: Sequence[float]) -> bool:
        for p_value, expected_p_value in zip(p_values, expected_p_values):
            if p_value < expected_p_value:
                return False

        return True

    # receives a quantum circuit, specifies which qubits should be measured and in which basis
    def get_measurement_configuration(self) -> MeasurementConfiguration:
        measurement_config = MeasurementConfiguration()
        for qubits, circ in [(self.qubits1, self.circuit1), (self.qubits2, self.circuit2)]:
            if "x" in self.basis:
                measurement_config.add_measurement(self.measurement_ids["x"], circ, {i: measure_x() for i in qubits})
            if "y" in self.basis:
                measurement_config.add_measurement(self.measurement_ids["y"], circ, {i: measure_y() for i in qubits})
            if "z" in self.basis:
                measurement_config.add_measurement(self.measurement_ids["z"], circ, {i: measure_z() for i in qubits})
        return measurement_config

    def get_measurements_from_circuits(self, measurements: Measurements) -> list:
        """
        Extract the measurement outcomes relevant to this assertion for both circuits.

        For each basis in self.basis, this method retrieves the full measurement counts for
        self.circuit1 and self.circuit2 using their respective measurement_ids. It then extracts
        only the bits corresponding to the qubits specified in self.qubits1 and self.qubits2.

        The extraction uses the convention that the bit for qubit i is:
            bitstring[len(bitstring) - i - 1]

        Counts for outcomes yielding the same extracted key are summed together.

        :param measurements: A Measurements object that holds all measurement results.
        :return: A dictionary organised by basis, with each basis containing a dict for each circuit.
        """
        results = []
        for basis in self.basis:
            # Process measurements for circuit1.
            full_counts1 = measurements.get_counts(self.circuit1, self.measurement_ids[basis])
            extracted_counts1 = {}
            for bitstring, count in full_counts1.items():
                # Extract only the bits corresponding to the qubits in self.qubits1.
                extracted_key = "".join(bitstring[len(bitstring) - q - 1] for q in self.qubits1)
                extracted_counts1[extracted_key] = extracted_counts1.get(extracted_key, 0) + count
            results.append(extracted_counts1)

            # Process measurements for circuit2.
            full_counts2 = measurements.get_counts(self.circuit2, self.measurement_ids[basis])
            extracted_counts2 = {}
            for bitstring, count in full_counts2.items():
                # Extract only the bits corresponding to the qubits in self.qubits2.
                extracted_key = "".join(bitstring[len(bitstring) - q - 1] for q in self.qubits2)
                extracted_counts2[extracted_key] = extracted_counts2.get(extracted_key, 0) + count
            results.append(extracted_counts2)

        return results
