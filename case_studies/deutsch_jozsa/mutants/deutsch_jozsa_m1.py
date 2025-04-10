from qiskit import QuantumCircuit


def deutsch_jozsa(oracle: QuantumCircuit) -> QuantumCircuit:
    circuit = QuantumCircuit(oracle.num_qubits, oracle.num_qubits)
    
    __qmutpy_qgi_func__(circuit, circuit.qubits[-1])  # mutation - equivalent
    for qubit in circuit.qubits:
        circuit.h(qubit)
        
    circuit.append(oracle, circuit.qubits)

    for qubit in circuit.qubits[:-1]:
        circuit.h(qubit)

    return circuit


def __qmutpy_qgi_func__(circ, qubit):
    circ.x(qubit)
    circ.id(qubit)
