# abstract base class for input_generators that creates an abstract method called generate
from abc import ABC, abstractmethod


class InputGenerator(ABC):
    @abstractmethod
    def generate(self, seed):
        pass

    # This clash (and all derived classes) need to be hashable for the input generator check to work
    def __hash__(self) -> int:
        # this hashes all class variables + the name of the class
        return hash((self.__class__.__name__,) + tuple(sorted(vars(self).items())))

    def __eq__(self, other):
        if isinstance(other, InputGenerator):
            return hash(self) == hash(other)
        return False
