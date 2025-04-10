# class that inherits from property based test
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators import RandomState, RandomUnitary
from case_studies.quantum_teleportation.quantum_teleportation import quantum_teleportation


class UnitaryBeforeAndAfterTeleport(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        state = RandomState(1)
        unitary = RandomUnitary(1, 1)
        return [state, unitary]

    # specify the preconditions for the test
    def preconditions(self, q0, unitary):
        return True

    # specify the operations to be performed on the input
    def operations(self, q0, unitary):
        # apply unitary on first qubit then teleport
        qc = QuantumCircuit(3, 3)
        qc.initialize(q0, [0])
        qc.append(unitary, [0])
        qt = quantum_teleportation()
        # stitch qc and quantum_teleportation together
        qc = qc.compose(qt)

        # apply teleport then apply unitary on third qubit
        qc2 = QuantumCircuit(3, 3)
        qc2.initialize(q0, [0])
        qt2 = quantum_teleportation()
        # stitch qc and quantum_teleportation together
        qc2 = qc2.compose(qt2)
        qc2.append(unitary, [2])

        self.statistical_analysis.assert_equal(self, [0, 1, 2], qc, [0, 1, 2], qc2)
