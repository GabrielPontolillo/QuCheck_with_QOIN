# create class that all property based tests inherit from, and add the required methods that specify input
# generation, preconditions, operations, and post conditions
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit

from qucheck.input_generators.input_generator import InputGenerator


class Property(ABC):
    # constructor to initialise the class with the statistical analysis object
    def __init__(self):
        self.statistical_analysis = None
        self.classical_assertion_outcome = True

    # generate inputs for the test
    # outputs a list of the generated inputs
    @abstractmethod
    def get_input_generators(self) -> list[InputGenerator]:
        # specify return type as list of input generators
        pass

    # specify the preconditions for the test
    @abstractmethod
    def preconditions(self, *inputs) -> bool:
        pass

    # the coordinator will pass all inputs in list order to the operations method
    # specify the operations to be performed on the input, assumed to be idempotent
    @abstractmethod
    def operations(self, *inputs) -> tuple[QuantumCircuit]:
        pass
