from typing import Sequence
from uuid import uuid4

from qucheck.utils import HashableQuantumCircuit
from qucheck.stats.assertion import StandardAssertion
from qucheck.stats.measurement_configuration import MeasurementConfiguration
from qucheck.stats.measurements import Measurements
from qucheck.stats.utils.common_measurements import measure_x, measure_y, measure_z


class AssertMostFrequent(StandardAssertion):
    def __init__(self, qubits: Sequence[int], circuit: HashableQuantumCircuit, states: Sequence[str], basis=["z"]) -> None:
        super().__init__()
        self.qubits = qubits
        self.circuit = circuit
        self.states = states
        self.basis = basis
        self.measurement_ids = {basis: uuid4() for basis in basis}

    def calculate_outcome(self, measurements: Measurements) -> bool:
        for basis in self.basis:
            counts = measurements.get_counts(self.circuit, self.measurement_ids[basis])
            # get the key with the largest counts, or frequency in the dictionary
            max_key = max(zip(counts.values(), counts.keys()))[1]

            # get only the qubits from the max string that we are checking
            relevant_bits_in_max_string = ""
            for qubit in self.qubits:
                relevant_bits_in_max_string += max_key[len(max_key) - qubit - 1]

            # check that most frequent state is in the states we are checking for current basis
            if relevant_bits_in_max_string not in self.states:
                return False

        return True

    def get_measurement_configuration(self) -> MeasurementConfiguration:
        measurement_config = MeasurementConfiguration()
        if "x" in self.basis:
            measurement_config.add_measurement(self.measurement_ids["x"], self.circuit, {i: measure_x() for i in self.qubits})
        if "y" in self.basis:
            measurement_config.add_measurement(self.measurement_ids["y"],self.circuit, {i: measure_y() for i in self.qubits})
        if "z" in self.basis:
            measurement_config.add_measurement(self.measurement_ids["z"],self.circuit, {i: measure_z() for i in self.qubits})

        return measurement_config

    def get_measurements_from_circuits(self, measurements: Measurements) -> list:
        """
        For each basis in self.basis, retrieve the full measurement counts from the Measurements object
        for self.circuit, then extract only the bits corresponding to self.qubits.
        The extraction uses the convention that the bit for qubit i is:
            bitstring[len(bitstring) - i - 1]
        Counts for outcomes yielding the same extracted key are summed.

        :param measurements: A Measurements object containing all measurement results.
        :return: A dictionary organised by basis containing the extracted counts.
        """
        results = []
        for basis in self.basis:
            full_counts = measurements.get_counts(self.circuit, self.measurement_ids[basis])
            extracted_counts = {}
            for bitstring, count in full_counts.items():
                extracted_key = "".join(bitstring[len(bitstring) - q - 1] for q in self.qubits)
                extracted_counts[extracted_key] = extracted_counts.get(extracted_key, 0) + count
            results.append(extracted_counts)
        return results
