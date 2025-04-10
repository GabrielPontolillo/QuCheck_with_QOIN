from typing import Sequence

from abc import ABC, abstractmethod
from uuid import uuid4

from qucheck.stats.measurement_configuration import MeasurementConfiguration
from qucheck.stats.measurements import Measurements


class Assertion(ABC):
    """super class of each assertion,
    hashable by reference (same object gives same hash but two equal objects will not - unless __hash__ overridden)
    """

    def __init__(self) -> None:
        super().__init__()
        self.id = hash(uuid4())

    failing_inputs = []

    @abstractmethod
    def get_measurement_configuration(self) -> MeasurementConfiguration:
        """get object containing information about which measurements to perform on which qubits

        Returns:
            MeasurementConfiguration: object containing information about which measurements to perform
        """
        pass

    def __hash__(self) -> int:
        return self.id


class StatisticalAssertion(Assertion):
    """super class of each assertion,
    hashable by reference (same object gives same hash but two equal objects will not - unless __hash__ overridden)
    """
    def __init__(self) -> None:
        super().__init__()
        self.id = hash(uuid4())

    @abstractmethod
    def calculate_p_values(self, measurements: Measurements) -> list[float]:
        """calculate p values for the assertion

        Args:
            measurements (Measurements): measurement outcomes
        Returns:
            list[float]: list of p values for this assertion
        """
        pass

    @abstractmethod
    def calculate_outcome(self, p_values: Sequence[float], expected_p_values: Sequence[float]) -> bool:
        """determine if assertion is true or not based on p values provided

        Args:
            p_values (Sequence[float]): sequence of p values, computed by calculate_p_values
            expected_p_values (Sequence[float]): sequence of p values, that is expected by statistics engine for the assertion to pass (respectively for each value in p_values)

        Returns:
            bool: true if assertion passed, false otherwise
        """
        pass


# TODO: Ideally there should be one superclass Assertion, that then has two subclasses StandardAssertion,
#  and StatisticalAssertion, with the distinction that one assertion will apply the calculate p_value e.t.c
# I will not care about this right now, I just want to get a more simple assertion to work that does not apply
# statistical tests, as this would be useful for the particular implementation to check for entanglement and assert_most_frequent
class StandardAssertion(Assertion):
    """super class of each assertion,
    hashable by reference (same object gives same hash but two equal objects will not - unless __hash__ overridden)
    """

    def __init__(self) -> None:
        super().__init__()
        self.id = hash(uuid4())

    @abstractmethod
    def calculate_outcome(self, measurements: Measurements) -> bool:
        """determine if assertion is true or not based on measurements only

        Args:
            measurements (Measurements): measurement outcomes

        Returns:
            bool: true if assertion passed, false otherwise
        """
        pass
