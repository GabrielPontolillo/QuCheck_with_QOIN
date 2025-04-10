import random
import unittest

from qiskit import QuantumCircuit

from qucheck.input_generators import RandomStatePreparationCircuit
from qucheck.test_runner import TestRunner
from tests.mock_properties.other_mocks import InOutQTCircuitGen, NotTeleportedPlusCircuitGen


class TestRandomStatePreparationCircuit(unittest.TestCase):
    def setUp(self):
        self.generator = RandomStatePreparationCircuit(2, 5)
        self.num_inputs = 10
        self.num_measurements = 10000

    def test_generates_circuit_instance(self):
        circuit = self.generator.generate(42)
        self.assertIsInstance(circuit, QuantumCircuit)

    def test_generates_circuit_within_qubit_range(self):
        circuit = self.generator.generate(42)
        self.assertGreaterEqual(circuit.num_qubits, 2)
        self.assertLessEqual(circuit.num_qubits, 5)

    def test_generates_deterministic_circuit(self):
        circuit1 = self.generator.generate(42)
        circuit2 = self.generator.generate(42)
        self.assertEqual(circuit1, circuit2)

    def test_first_gate_is_a_u_gate(self):
        circuit = self.generator.generate(42)
        self.assertEqual(circuit.data[0][0].name, 'u')

    def test_last_gate_is_a_u_gate(self):
        circuit = self.generator.generate(42)
        self.assertEqual(circuit.data[-1][0].name, 'u')

    def test_cnots_are_applied(self):
        circuit = self.generator.generate(41)
        cnots = [gate for gate in circuit.data if gate[0].name == 'cx']
        self.assertGreater(len(cnots), 0)

    def test_cnots_not_applied(self):
        circuit = self.generator.generate(42)
        cnots = [gate for gate in circuit.data if gate[0].name == 'cx']
        self.assertEqual(len(cnots), 0)

    def test_cnots_not_more_than_qubits(self):
        val = random.randint(0, 120194)
        circuit = self.generator.generate(val)
        cnots = [gate for gate in circuit.data if gate[0].name == 'cx']
        self.assertLessEqual(len(cnots), circuit.num_qubits)

    def test_number_of_u_equal_to_two_times_number_of_qubits(self):
        val = random.randint(0, 120194)
        circuit = self.generator.generate(val)
        us = [gate for gate in circuit.data if gate[0].name == 'u']
        self.assertEqual(len(us), 2 * circuit.num_qubits)

    def test_qt_in_out_still_passes_with_this_gen(self):
        runner = TestRunner([InOutQTCircuitGen], self.num_inputs, 42, self.num_measurements)
        runner.run_tests()
        self.assertEqual(runner.list_passing_properties(), [InOutQTCircuitGen])

    def test_qt_plus_still_passes_with_this_gen(self):
        runner = TestRunner([NotTeleportedPlusCircuitGen], self.num_inputs, 42, self.num_measurements)
        runner.run_tests()
        self.assertEqual(runner.list_passing_properties(), [NotTeleportedPlusCircuitGen])
