from time import time
from qiskit import QuantumCircuit
from qucheck.stats.measurement_configuration import MeasurementConfiguration
from qucheck.utils import HashableQuantumCircuit


class CircuitGenerator:
    def __init__(self, run_optimization=True) -> None:
        self.run_optimization = run_optimization
        self.measurement_info_for_unique_circuits: dict[HashableQuantumCircuit, list[tuple[str, HashableQuantumCircuit]]] = {}
        self.unoptimized_measurement_info: dict[HashableQuantumCircuit, list[tuple[str, dict[int, QuantumCircuit]]]] = {}
    
    def add_measurement_configuration(self, measurement_config: MeasurementConfiguration) -> None:
        base_circuits = measurement_config.get_measured_circuits()
        for circuit in base_circuits:
            if circuit in self.unoptimized_measurement_info:
                for measurement_id, qubits_measurements in measurement_config.get_measurements_for_circuit(circuit):
                    self.unoptimized_measurement_info[circuit].append((measurement_id, qubits_measurements))
            else:
                self.unoptimized_measurement_info[circuit] = measurement_config.get_measurements_for_circuit(circuit)
    
    def _optimize(self) -> list[HashableQuantumCircuit]:
        # get unique base circuits:
        base_circuits = set(self.unoptimized_measurement_info.keys())
        # print(len(base_circuits), [len(c.data) for c in base_circuits])
        # since we hash by reference, keep track of all duplicated circuits to feed back in get_measurement_info
        for unique_circuit in base_circuits:
            self._get_full_circuits(unique_circuit)
        # remove duplicates
        full_unique_circuits = set(self.measurement_info_for_unique_circuits.keys())
        return full_unique_circuits
    
    def _get_unoptimized_circuits(self) -> list[QuantumCircuit]:
        base_circuits = list(self.unoptimized_measurement_info.keys())
        full_circuits = []
        for circuit in base_circuits:
            for measurement_id, qubit_spec in self.unoptimized_measurement_info[circuit]:
                qc = circuit.copy()
                for qubit, measurement in qubit_spec.items():
                    qc.compose(measurement, (qubit,), (qubit,), inplace=True)
                qc = self._get_hashable_circuit(qc)
                self.measurement_info_for_unique_circuits[qc] = [(measurement_id, circuit)]
                full_circuits.append(qc)
        return full_circuits

    def _get_full_circuits(self, circuit: HashableQuantumCircuit):
        all_measurement_specifications = []
        for measurement_id, qubit_spec in self.unoptimized_measurement_info[circuit]:
            all_measurement_specifications.append((measurement_id, qubit_spec, circuit))
        measurement_specification_inserted = [False for _ in all_measurement_specifications]
        while True:
            qc = circuit.copy()
            inserted_qubits = {}
            measurement_specifications_in_circuit = []
            if False not in measurement_specification_inserted:
                return
            
            for i in range(len(all_measurement_specifications)):
                if measurement_specification_inserted[i]:
                    continue

                measurement_id, qubit_measurement_map, original_circuit = all_measurement_specifications[i]
                
                overlapping_qubits = set(inserted_qubits.keys()).intersection(qubit_measurement_map.keys())
                should_append_circuit = True
                for qubit in overlapping_qubits:
                    if inserted_qubits[qubit] != qubit_measurement_map[qubit]:
                        should_append_circuit = False
                        break
                
                if not should_append_circuit:
                    continue
                else:
                    measurement_specifications_in_circuit.append((measurement_id, original_circuit))
                    measurement_specification_inserted[i] = True
                    for qubit, measurement in qubit_measurement_map.items():
                        if qubit not in inserted_qubits:
                            qc.compose(measurement, (qubit,), (qubit,), inplace=True)
                    inserted_qubits.update(qubit_measurement_map)

            self.measurement_info_for_unique_circuits[qc] = measurement_specifications_in_circuit

    def get_circuits_to_execute(self) -> list[QuantumCircuit]:
        """
        Returns:
            list[QuantumCircuit]: list of unique circuits to be executed based on all measurement configs added to optimizer so far
        """
        if self.run_optimization:
            return [self._get_executable_circuit(circuit) for circuit in self._optimize()]
        else:
            return [self._get_executable_circuit(circuit) for circuit in self._get_unoptimized_circuits()]

    def get_measurement_info(self, circuit: QuantumCircuit) -> list[tuple[str, HashableQuantumCircuit]]:
        """
        Args:
            circuit (QuantumCircuit): one of circuits returned by get_circuits_to_execute

        Returns:
            list[tuple[str, HashableQuantumCircuit]]: list of tuples (measurement_id, original circuit) 
                - data from measurement configuration provided to the optimizer
        """
        return self.measurement_info_for_unique_circuits[self._get_hashable_circuit(circuit)]
    
    def _get_hashable_circuit(self, circuit: QuantumCircuit) -> HashableQuantumCircuit:
        circuit.__class__ = HashableQuantumCircuit
        return circuit
    
    def _get_executable_circuit(self, circuit: HashableQuantumCircuit) -> QuantumCircuit:
        circuit.__class__ = QuantumCircuit
        return circuit
