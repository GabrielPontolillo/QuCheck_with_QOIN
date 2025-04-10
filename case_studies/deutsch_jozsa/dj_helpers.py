import random
from qiskit import QuantumCircuit
from qucheck.input_generators.input_generator import InputGenerator


class BalancedOracleInputGenerator(InputGenerator):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def generate(self, seed):
        random.seed(seed)
        num_qubits = random.randint(self.low, self.high)
        circuit = QuantumCircuit(num_qubits)

        # make a random array of 0s and 1s
        xs = [random.randint(0, 1) for _ in range(num_qubits - 1)]

        # random array of 0s and 1s, but we must guarantee at least one 1
        if num_qubits > 1:
            rand_bits = [int(i) for i in bin(random.randint(1, 2**(num_qubits - 1) - 1))[2:]]
            cxs = [0]*((num_qubits - 1) - len(rand_bits)) + rand_bits
        else:
            cxs = []

        for idx, i in enumerate(xs):
            if i == 1:
                circuit.x(circuit.qubits[idx])

        for idx, i in enumerate(cxs):
            if i == 1:
                circuit.cx(circuit.qubits[idx], circuit.qubits[-1])

        for idx, i in enumerate(xs):
            if i == 1:
                circuit.x(circuit.qubits[idx])

        return circuit


class ConstantOracleInputGenerator(InputGenerator):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def generate(self, seed):
        random.seed(seed)
        num_qubits = random.randint(self.low, self.high)

        # randomly choose between constant 0 and constant 1
        constant = random.randint(0, 1)

        qc = QuantumCircuit(num_qubits)

        if constant == 1:
            qc.x(qc.qubits[-1])

        return qc


class RandomOracleInputGenerator(InputGenerator):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def generate(self, seed):
        random.seed(seed)
        # randomly choose between constant 0 and constant 1
        oracle_choice = random.randint(0, 1)

        if oracle_choice == 1:
            return ConstantOracleInputGenerator(self.low, self.high).generate(seed)
        else:
            return BalancedOracleInputGenerator(self.low, self.high).generate(seed)


# defines a function that vertically merges two circuits, so that the first circuit is on top of the second, with the
# last qubit of the first circuit being connected to the last qubit of the second circuit
def vmerge(circuit1, circuit2):
    # as one register is merged, we need to subtract by one
    new_size = circuit1.num_qubits + circuit2.num_qubits - 1

    qc = QuantumCircuit(new_size)
    qc.compose(circuit1, list(range(0, circuit1.num_qubits - 1)) + [new_size - 1], inplace=True)

    # need to start after the first circuit ends
    qc.compose(circuit2, list(range(circuit1.num_qubits - 1, new_size - 1)) + [new_size - 1], inplace=True)

    return qc

