from qiskit import QuantumCircuit


# returns grover's algorithm on the provided oracle
def grovers_algorithm(grover_oracle: QuantumCircuit, iterations: int):

    num_qubits = grover_oracle.num_qubits

    grover_qc = QuantumCircuit(num_qubits, num_qubits)

    # initialise lower register to |1>
    grover_qc.x(num_qubits - 1)

    # initialise superposition to get |++....++>|->
    grover_qc.h(range(num_qubits))



    # apply the oracle and reflection about the mean
    for i in range(iterations):
        grover_qc.compose(grover_oracle, range(num_qubits), inplace=True)



        # reflection about the mean
        grover_qc.h(range(num_qubits - 1))
        grover_qc.y(range(num_qubits - 1))  # here
        grover_qc.h(num_qubits - 2)
        grover_qc.mcx(list(range(num_qubits - 2)), num_qubits - 2)
        grover_qc.h(num_qubits - 2)
        grover_qc.x(range(num_qubits - 1))
        grover_qc.h(range(num_qubits - 1))



    return grover_qc
