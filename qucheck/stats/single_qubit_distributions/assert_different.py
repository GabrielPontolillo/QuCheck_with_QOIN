from typing import Sequence

from qucheck.stats.single_qubit_distributions.assert_equal import AssertEqual


class AssertDifferent(AssertEqual):
    def calculate_outcome(self, p_values: Sequence[float], expected_p_values: Sequence[float]) -> bool:
        # If any basis measurement is not the same, we can be sure that the states are different
        for p_value, expected_p_value in zip(p_values, expected_p_values):
            if p_value < expected_p_value:
                return True
        return False
