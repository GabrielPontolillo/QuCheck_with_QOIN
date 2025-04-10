from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


def deutsch_jozsa(oracle: QuantumCircuit) -> QuantumCircuit:
    circuit = QuantumCircuit(oracle.num_qubits, oracle.num_qubits)
    
    circuit.x(circuit.qubits[-1])
    for qubit in circuit.qubits:
        circuit.h(qubit)
        
    circuit.append(oracle, circuit.qubits)

    for qubit in circuit.qubits[:-1]:
        circuit.h(qubit)

    # semantic preserving changes loc 5, gates 7, index 0
    circuit.x(0)
    circuit.y(0)
    circuit.append(Pauli('iZ'), [0])

    return circuit

