from qiskit import QuantumCircuit


def measure_y() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.sdg(0)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def measure_z() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    return circuit


def measure_x() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def ry_then_measure(theta) -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)
    return circuit

