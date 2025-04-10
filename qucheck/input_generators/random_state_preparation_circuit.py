import random

from qiskit import QuantumCircuit

from qucheck.input_generators.input_generator import InputGenerator


# uses the technique from "Quratest" to generate a circuit to randomly initialise a state using U gates and CNOT gates
# https://doi.org/10.1109/ASE56229.2023.00196

class RandomStatePreparationCircuit(InputGenerator):
    def __init__(self, number_of_qubits_low, number_of_qubits_high):
        self.number_of_qubits_low = number_of_qubits_low
        self.number_of_qubits_high = number_of_qubits_high

    def generate(self, seed) -> QuantumCircuit:
        random.seed(seed)
        number_of_qubits = random.randint(self.number_of_qubits_low, self.number_of_qubits_high)

        qc = QuantumCircuit(number_of_qubits)

        # apply random u gates to all qubits
        for i in range(number_of_qubits):
            qc.u(random.random() * 2 * 3.14159, random.random() * 3.14159, random.random() * 2 * 3.14159, i)

        # randomly apply random cnot gates
        controls = random.sample(range(number_of_qubits), random.randint(0, number_of_qubits))
        if len(controls) > 0 and number_of_qubits > 1:
            for control in controls:
                target = random.choice([x for x in range(number_of_qubits) if x != control])
                qc.cx(control, target)

        # apply random u gates to all qubits
        for i in range(number_of_qubits):
            qc.u(random.random() * 2 * 3.14159, random.random() * 3.14159, random.random() * 2 * 3.14159, i)

        return qc
