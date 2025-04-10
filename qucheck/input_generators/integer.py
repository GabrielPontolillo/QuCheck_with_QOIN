from qucheck.input_generators.input_generator import InputGenerator
import random


class Integer(InputGenerator):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def generate(self, seed) -> int:
        random.seed(seed)
        return random.randint(self.low, self.high)
