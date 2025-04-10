from qiskit import QuantumCircuit

from qucheck.utils import HashableQuantumCircuit


class MeasurementConfiguration:
    """
    A class to configure and manage measurement specifications for quantum circuits.

    Attributes:
        _data (dict[HashableQuantumCircuit, list[tuple[str, dict[int, QuantumCircuit]]]]):
            A mapping of hashable quantum circuits to a list of measurement specifications.
            Each specification includes a measurement ID and a dictionary mapping register indices
            to circuits used for basis transformation.
    """

    def __init__(self) -> None:
        """
        Initializes an empty MeasurementConfiguration instance.
        """
        self._data: dict[HashableQuantumCircuit, list[tuple[str, dict[int, QuantumCircuit]]]] = {}

    def add_measurement(self, measurement_id: str, circuit: QuantumCircuit, measurement_specification: dict[int, QuantumCircuit]) -> None:
        """
        Adds a measurement specification to the configuration.

        Args:
            measurement_id (str):
                An identifier for the measurement, indicating the basis ("x", "y", "z")
                or any other intended basis transformation.
            circuit (QuantumCircuit):
                The quantum circuit to measure.
            measurement_specification (dict[int, QuantumCircuit]):
                A dictionary specifying which registers to measure and the circuits
                to append for basis transformations.
        """
        if circuit in self._data:
            self._data[circuit].append((measurement_id, measurement_specification))
        else:
            self._data[circuit] = [(measurement_id, measurement_specification)]

    def get_measured_circuits(self) -> tuple[HashableQuantumCircuit]:
        """
        Retrieves all quantum circuits with associated measurement specifications.

        Returns:
            tuple[HashableQuantumCircuit]:
                A tuple of quantum circuits with measurements configured.
        """
        return tuple(self._data.keys())

    def get_measurements_for_circuit(self, circuit: HashableQuantumCircuit) -> list[tuple[str, dict[int, QuantumCircuit]]]:
        """
        Retrieves the measurement specifications for a given quantum circuit.

        Args:
            circuit (HashableQuantumCircuit):
                The quantum circuit whose measurements are to be retrieved.

        Returns:
            list[tuple[str, dict[int, QuantumCircuit]]]:
                A list of measurement specifications for the given circuit.
                Each specification includes the measurement ID and a dictionary mapping
                register indices to basis transformation circuits.
        """
        return self._data[circuit]