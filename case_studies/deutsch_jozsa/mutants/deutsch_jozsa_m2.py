from qiskit import QuantumCircuit


def deutsch_jozsa(oracle: QuantumCircuit) -> QuantumCircuit:
    circuit = QuantumCircuit(oracle.num_qubits, oracle.num_qubits)
    
    circuit.x(circuit.qubits[-1])
    for qubit in circuit.qubits:
        circuit.h(qubit)
        
    circuit.append(oracle, circuit.qubits)

    for qubit in circuit.qubits[:-1]:
        __qmutpy_qgi_func__(circuit, qubit)

    return circuit


def __qmutpy_qgi_func__(circ, qubit):
    circ.h(qubit)
    circ.tdg(qubit)
