# class that inherits from property based test
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators import RandomStatePreparationCircuit, Integer
from case_studies.quantum_teleportation.quantum_teleportation import quantum_teleportation
from case_studies.superdense_coding.superdense_coding import superdense_coding


class InOutQTCircuitGen(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        circuit = RandomStatePreparationCircuit(1, 1)
        return [circuit]

    # specify the preconditions for the test
    def preconditions(self, circuit):
        return True

    # specify the operations to be performed on the input
    def operations(self, circuit):
        qc = QuantumCircuit(3, 3)
        qc.compose(circuit, [0], inplace=True)
        qt = quantum_teleportation()
        # stitch qc and quantum_teleportation together
        qc = qc.compose(qt)

        # initialise qubit to compare to:
        qc2 = QuantumCircuit(1, 1)
        qc2.compose(circuit, [0], inplace=True)
        self.statistical_analysis.assert_equal(self, 2, qc, 0, qc2)


class NotTeleportedPlusCircuitGen(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        circuit = RandomStatePreparationCircuit(1, 1)
        return [circuit]

    # specify the preconditions for the test
    def preconditions(self, q0):
        return True

    # specify the operations to be performed on the input
    def operations(self, circuit):
        # breaks if classical register is not explicitly defined?
        qc = QuantumCircuit(3, 3)
        qc.compose(circuit, [0], inplace=True)
        qt = quantum_teleportation()
        # stitch qc and quantum_teleportation together
        qc = qc.compose(qt)

        # initialise another circuit to |++> state
        # breaks if classical register is not explicitly defined?
        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.h(1)

        self.statistical_analysis.assert_equal(self, [0, 1], qc, [0, 1], qc2)


class IncorrectBitsSent(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        b1 = Integer(0, 1)
        b2 = Integer(0, 1)
        return [b1, b2]

    # specify the preconditions for the test
    def preconditions(self, b1, b2):
        return b1 != b2

    # specify the operations to be performed on the input
    def operations(self, b1, b2):
        qc = QuantumCircuit(2, 2)
        sd = superdense_coding(b1, b2)
        # stitch qc and quantum_teleportation together
        qc = qc.compose(sd)

        # initialise qubit to compare to:
        qc2 = QuantumCircuit(2, 2)

        # flip ordering, should cause failure
        if b1 == 1:
            qc2.x(1)
        if b2 == 1:
            qc2.x(0)

        self.statistical_analysis.assert_equal(self, [0, 1], qc, [0, 1], qc2)


