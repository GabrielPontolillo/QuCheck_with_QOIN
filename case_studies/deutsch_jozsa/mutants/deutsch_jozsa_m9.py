from qiskit import QuantumCircuit


def deutsch_jozsa(oracle: QuantumCircuit) -> QuantumCircuit:
    circuit = QuantumCircuit(oracle.num_qubits, oracle.num_qubits)
    
    circuit.x(circuit.qubits[-1])
    for qubit in circuit.qubits:
        circuit.h(qubit)
        
    circuit.append(oracle, circuit.qubits)

    for qubit in circuit.qubits[:-1]:
        circuit.id(qubit)  # mutation

    return circuit

