# class that inherits from property based test
from qiskit import QuantumCircuit
from qucheck.property import Property
from qucheck.input_generators import RandomState, RandomStatePreparationCircuit, RandomPauliBasisState


class EntangledPrecondition(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        # The assumption of our code is that we need at least one input generator, if we do not include it,
        # the code breaks in the test runner - also its a fine assumption to include, property based testing requires inputs
        return [RandomState(2)]

    # specify the preconditions for the test
    def preconditions(self, mock):
        return True

    # specify the operations to be performed on the input
    def operations(self, mock):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        self.statistical_analysis.assert_entangled(self, [0, 1], qc)


class EntangledCheckOnUnentangledState(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        # The assumption of our code is that we need at least one input generator, if we do not include it,
        # the code breaks in the test runner - also its a fine assumption to include, property based testing requires inputs
        return [RandomState(2)]

    # specify the preconditions for the test
    def preconditions(self, mock):
        return True

    # specify the operations to be performed on the input
    def operations(self, mock):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        self.statistical_analysis.assert_entangled(self, [0, 1], qc)


class EntangledCheckOnGHZState(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        # The assumption of our code is that we need at least one input generator, if we do not include it,
        # the code breaks in the test runner - also its a fine assumption to include, property based testing requires inputs
        return [RandomPauliBasisState()]

    # specify the preconditions for the test
    def preconditions(self, mock):
        return True

    # specify the operations to be performed on the input
    def operations(self, mock):
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        self.statistical_analysis.assert_entangled(self, [0, 1, 2], qc, basis=["z"])


class EntangledCheckOnRandomState(Property):
    # specify the inputs that are to be generated
    def get_input_generators(self):
        # The assumption of our code is that we need at least one input generator, if we do not include it,
        # the code breaks in the test runner - also its a fine assumption to include, property based testing requires inputs
        return [RandomPauliBasisState(3, 3, ('z'))]

    # specify the preconditions for the test
    def preconditions(self, mock):
        return True

    # specify the operations to be performed on the input
    def operations(self, mock):
        qc = QuantumCircuit(3, 3)
        # qc.initialize(mock, [0, 1, 2])
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        qc2 = QuantumCircuit(3, 3)
        qc2.
        self.statistical_analysis.assert_entangled(self, [0, 1, 2], qc)
