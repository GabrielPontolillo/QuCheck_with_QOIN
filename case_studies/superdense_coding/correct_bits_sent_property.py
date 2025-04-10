# class that inherits from property based test
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators import Integer
from case_studies.superdense_coding.superdense_coding import superdense_coding


class CorrectBitsSent(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        b1 = Integer(0, 1)
        b2 = Integer(0, 1)
        return [b1, b2]

    # specify the preconditions for the test
    def preconditions(self, b1, b2):
        return True

    # specify the operations to be performed on the input
    def operations(self, b1, b2):
        qc = QuantumCircuit(2, 2)
        sd = superdense_coding(b1, b2)
        # stitch qc and quantum_teleportation together
        qc = qc.compose(sd)

        print(qc)

        print(b1)
        print(b2)

        # initialise qubit to compare to:
        qc2 = QuantumCircuit(2, 2)
        if b1 == 1:
            qc2.x(0)
        if b2 == 1:
            qc2.x(1)

        # self.statistical_analysis.assert_equal(self, [0, 1], qc, [0, 1], qc2)
        self.statistical_analysis.assert_equal(self, [0, 1], qc, [0, 1], qc2)
