from qucheck.utils import HashableQuantumCircuit


class Measurements:
    """
    A class to store and manage measurement results for quantum circuits.

    Structure:
        - Key: The base quantum circuit that has been measured.
        - Value: A dictionary where:
            - Key: Measurement name (e.g., the basis "x", "y", "z").
            - Value: A dictionary of bitstrings and their counts.

    Example:
        (circuit): { "x": {"00": 10, "01": 15}, "y": {"10": 8, "11": 20} }
    """
    def __init__(self) -> None:
        """
        Initializes an empty Measurements instance.
        """
        self._data: dict[HashableQuantumCircuit, dict[str, dict[str, int]]] = {}

    def add_measurement(self, circuit: HashableQuantumCircuit, measurement_id: str, counts: dict[str, int]) -> None:
        """
        Adds measurement results for a given circuit and basis.

        Args:
            circuit (HashableQuantumCircuit):
                The quantum circuit that was measured.
            measurement_id (str):
                The name of the measurement basis (e.g., "x", "y", "z").
            counts (dict[str, int]):
                A dictionary of bitstrings and their counts from the measurement.
        """
        if circuit in self._data:
            self._data[circuit][measurement_id] = counts
        else:
            self._data[circuit] = {measurement_id: counts}

    def get_counts(self, circuit: HashableQuantumCircuit, measurement_id: str) -> dict[str, int]:
        """
        Retrieves the counts for a specific circuit and measurement basis.

        Args:
            circuit (HashableQuantumCircuit):
                The quantum circuit for which counts are being retrieved.
            measurement_id (str):
                The name of the measurement basis.

        Returns:
            dict[str, int]:
                A dictionary of bitstrings and their counts for the specified measurement.
        """
        return self._data[circuit][measurement_id]