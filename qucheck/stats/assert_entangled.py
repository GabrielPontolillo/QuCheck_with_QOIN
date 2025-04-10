from typing import Sequence
from uuid import uuid4

from qucheck.utils import HashableQuantumCircuit
from qucheck.stats.assertion import StandardAssertion
from qucheck.stats.measurement_configuration import MeasurementConfiguration
from qucheck.stats.measurements import Measurements
from qucheck.stats.utils.common_measurements import measure_x, measure_y, measure_z


#  This should work for simple things, but turns out that asserting that qubits are entangled is actually a more
#  complex problem than what I first thought, I think the best way to go about this would be to allow the user
#  to specify which basis they want to check the entanglement in, and then we can check the entanglement in that basis
#  currently we can only check three basis, but in the future we can allow the user to specify any basis they want
#  - (in particular, theyd need to provide a gate that rotates from their entangled basis to Z basis, and then we can measure in Z basis, king of like what we do for X, and Y)
#  which will change the measurement configuration, and then we can check the entanglement in that basis
#  if that Is implemented it should just work out of the box with the current implementation

#  going in the direction of CHSH inequality, works only for 2 qubits, extending to more qubits is possible through other
#  metrics, but would balloon the number of measurements needed
class AssertEntangled(StandardAssertion):
    def __init__(self, qubits: Sequence[int], circuit: HashableQuantumCircuit, basis=["z"]) -> None:
        super().__init__()
        self.qubits = qubits
        self.circuit = circuit
        self.basis = basis
        self.measurement_ids = {basis: uuid4() for basis in basis}

    def calculate_outcome(self, measurements: Measurements) -> bool:
        for basis in self.basis:
            counts = measurements.get_counts(self.circuit, self.measurement_ids[basis])
            # we know check that if 00 is in keys, then there must only be 11 in the keys
            # and the other way around  (01, 10)
            bitstrings = counts.keys()
            print(bitstrings)
            relevant_bitstrings = []

            if len(bitstrings) != 2:
                return False
            else:
                # get the relevant bits from the bitstrings only
                for bitstring in bitstrings:
                    string_builder = ""
                    for qubit in self.qubits:
                        string_builder += bitstring[len(bitstring) - qubit - 1]
                    relevant_bitstrings.append(string_builder)

            # the ''.join flips the 1's and 0's in the bitstring, as if the state is entangled, then the two options are
            # 00 and 11, and the same for 01 and 10
            print(relevant_bitstrings)
            if relevant_bitstrings[0] != ''.join('1' if x == '0' else '0' for x in relevant_bitstrings[1]):
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


def extract_counts(bitstring: str, qubit1: int, qubit2: int, counts: dict[str, int]) -> int:
    """Extracts the counts of a specific bitstring using the specific provided qubits from a dictionary of counts
    Args:
        bitstring (str): The bitstring to extract
        qubit1 (int): The first qubit to check
        qubit2 (int): The second qubit to check
        counts (dict[str, int]): The dictionary of counts

    Returns:
        int: The number of counts for the given bitstring

    """
    total = 0
    for bits in counts:
        if bits[::-1][qubit1] == bitstring[0] and bits[::-1][qubit2] == bitstring[1]:
            total += counts.get(bits, 0)
    return total
