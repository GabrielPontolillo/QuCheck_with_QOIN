from qiskit import QuantumCircuit
import random

from qucheck.input_generators import InputGenerator


class RandomGroversOracleMarkedStatesPairGenerator(InputGenerator):
    def __init__(self, low: int, high: int, marked_states="random"):
        if low < 2:
            raise ValueError("Low must be greater than or equal to 2")
        self.low = low
        self.high = high
        self.marked_states = marked_states

    # generate a random grover's oracle with less than half of the states marked
    def generate(self, seed):
        random.seed(seed)
        num_qubits = random.randint(self.low, self.high)
        circuit = QuantumCircuit(num_qubits)

        N = 2**(num_qubits-1)
        if self.marked_states == "random":
            # choose to generate a random number of marked states, guaranteed to be less than 1/2 N states
            M = random.randint(1, (N//2)-1)
        elif self.marked_states == "too_many":
            M = random.randint((N//2)+1, N-1)
        else:
            M = self.marked_states

        # chooses M random states to mark from the set of all states that can be marked with the given number of qubits
        marked_states = tuple(random.sample(range(N - 1), M))

        for state in marked_states:
            marking_circ = marking_circuit(num_qubits, state)
            circuit = circuit.compose(marking_circ)
        # print("oracle (seed, num qubits, num marked states):", seed, num_qubits, M)
        return circuit, marked_states


# this is under the assumption that the MSB is q_0
def marking_circuit(num_qubits: int, integer_to_mark: int):
    assert 0 <= integer_to_mark < 2 ** (num_qubits-1), f"tried to mark a state :{integer_to_mark} that is larger than: {2**(num_qubits-1) - 1} for the number of upper register qubits {num_qubits-1}"

    circuit = QuantumCircuit(num_qubits)
    # print(integer_to_mark)
    binary = bin(integer_to_mark)[2:]
    binary = '0' * (num_qubits - 1 - len(binary)) + binary
    binary = binary[::-1]

    for idx, bit in enumerate(binary):
        if bit == '0':
            circuit.x(idx)

    # control over all qubits in the circuit (minus workspace register)
    circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)

    # uncompute the marked state
    for idx, bit in enumerate(binary):
        if bit == '0':
            circuit.x(idx)

    return circuit

