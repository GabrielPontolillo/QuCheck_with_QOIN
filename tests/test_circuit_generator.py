from unittest import TestCase

from qucheck.stats.circuit_generator import CircuitGenerator
from qucheck.stats.measurement_configuration import MeasurementConfiguration
from qucheck.stats.utils.common_measurements import measure_x, measure_y
from qucheck.utils import HashableQuantumCircuit


class TestCircuitGenerator(TestCase):
    def test_different_circuits_with_same_measurements_on_same_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        circ1 = HashableQuantumCircuit(2, 2)
        circ1.x(0)
        measurement1.add_measurement("1", circ1, {0: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        circ2 = HashableQuantumCircuit(2, 2)
        circ2.h(0)
        measurement2.add_measurement("2", circ2, {0: measure_x()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 2

    def test_different_circuits_with_diff_measurements_on_diff_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        circ1 = HashableQuantumCircuit(2, 2)
        circ1.x(0)
        measurement1.add_measurement("1", circ1, {0: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        circ2 = HashableQuantumCircuit(2, 2)
        circ2.h(0)
        measurement2.add_measurement("2", circ2, {1: measure_y()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 2

    def test_different_circuits_with_same_measurements_on_diff_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        circ1 = HashableQuantumCircuit(2, 2)
        circ1.x(0)
        measurement1.add_measurement("1", circ1, {0: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        circ2 = HashableQuantumCircuit(2, 2)
        circ2.h(0)
        measurement2.add_measurement("2", circ2, {1: measure_x()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 2

    def test_different_circuits_with_diff_measurements_on_same_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        circ1 = HashableQuantumCircuit(2, 2)
        circ1.x(0)
        measurement1.add_measurement("1", circ1, {0: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        circ2 = HashableQuantumCircuit(2, 2)
        circ2.h(0)
        measurement2.add_measurement("2", circ2, {0: measure_y()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 2

    def test_different_circuits_with_same_measurements_on_overlapping_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        circ1 = HashableQuantumCircuit(3, 3)
        circ1.x(0)
        measurement1.add_measurement("1", circ1, {0: measure_x(), 1: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        circ2 = HashableQuantumCircuit(3, 3)
        circ2.h(0)
        measurement2.add_measurement("2", circ2, {1: measure_x(), 2: measure_x()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 2

    def test_different_circuits_with_diff_measurements_on_overlapping_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        circ1 = HashableQuantumCircuit(3, 3)
        circ1.x(0)
        measurement1.add_measurement("1", circ1, {0: measure_x(), 1: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        circ2 = HashableQuantumCircuit(3, 3)
        circ2.h(0)
        measurement2.add_measurement("2", circ2, {1: measure_y(), 2: measure_x()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 2

    def test_same_circuits_with_same_measurements_on_same_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        measurement1.add_measurement("1", HashableQuantumCircuit(2, 2), {0: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        measurement2.add_measurement("2", HashableQuantumCircuit(2, 2), {0: measure_x()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 1

    def test_same_circuit_same_measurements_on_diff_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        measurement1.add_measurement("1", HashableQuantumCircuit(2, 2), {0: measure_x()})
        optimizer.add_measurement_configuration(measurement1)
        measurement2 = MeasurementConfiguration()
        measurement2.add_measurement("2", HashableQuantumCircuit(2, 2), {1: measure_x()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 1

    def test_same_circuit_different_measurements_on_diff_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        measurement1.add_measurement("1", HashableQuantumCircuit(2, 2), {0: measure_x()})
        optimizer.add_measurement_configuration(measurement1)

        measurement2 = MeasurementConfiguration()
        measurement2.add_measurement("2", HashableQuantumCircuit(2, 2), {1: measure_y()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 1

    def test_same_circuit_different_measurements_on_overlapping_qubits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        measurement1.add_measurement("1", HashableQuantumCircuit(3, 3), {0: measure_x(), 1: measure_x()})
        optimizer.add_measurement_configuration(measurement1)

        measurement2 = MeasurementConfiguration()
        measurement2.add_measurement("2", HashableQuantumCircuit(3, 3), {1: measure_y(), 2: measure_y()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 2

    def test_same_circuit_overlapping_qubit_measurements_with_overlap_with_equal_circuits(self):
        optimizer = CircuitGenerator()
        measurement1 = MeasurementConfiguration()
        measurement1.add_measurement("1", HashableQuantumCircuit(3, 3), {0: measure_x(), 1: measure_x()})
        optimizer.add_measurement_configuration(measurement1)

        measurement2 = MeasurementConfiguration()
        measurement2.add_measurement("2", HashableQuantumCircuit(3, 3), {1: measure_x(), 2: measure_y()})
        optimizer.add_measurement_configuration(measurement2)

        assert len(optimizer.get_circuits_to_execute()) == 1
