# class that inherits from property based test
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators import RandomState
from case_studies.quantum_teleportation.quantum_teleportation import quantum_teleportation


class Inq0EqualOutq2(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        state = RandomState(1)
        return [state]

    # specify the preconditions for the test
    def preconditions(self, q0):
        return True

    # specify the operations to be performed on the input
    def operations(self, q0):
        qc = QuantumCircuit(3, 3)
        qc.initialize(q0, [0])
        qt = quantum_teleportation()
        # stitch qc and quantum_teleportation together
        qc = qc.compose(qt)

        # initialise qubit to compare to:
        qc2 = QuantumCircuit(1, 1)
        qc2.initialize(q0, [0])
        self.statistical_analysis.assert_equal(self, 2, qc, 0, qc2)
