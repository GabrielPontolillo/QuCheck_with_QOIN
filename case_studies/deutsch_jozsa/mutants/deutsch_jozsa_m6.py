from qiskit import QuantumCircuit


def deutsch_jozsa(oracle: QuantumCircuit) -> QuantumCircuit:
    circuit = QuantumCircuit(oracle.num_qubits, oracle.num_qubits)
    
    circuit.h(circuit.qubits[-1])  # mutation
    for qubit in circuit.qubits:
        circuit.t(qubit)  # mutation
        
    circuit.append(oracle, circuit.qubits)

    for qubit in circuit.qubits[:-1]:
        circuit.h(qubit)

    return circuit

