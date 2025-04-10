from time import time
import numpy
from qiskit import QuantumCircuit

"""
hack to use quantum circuits as dictionary keys, 
this is hash by "reference" not by value - as in, two equal objects will not be hashed to the same value
but it is guaranteed that the same object will always return same hash value
"""


class HashableQuantumCircuit(QuantumCircuit):
    def __hash__(self):
        if hasattr(self, "_hash_val"):
            return self._hash_val
        else:
            self._hash_val = hash(tuple(hash_instruction(instruction) for instruction in self._data))
            return self._hash_val
        
    def copy(self):
        qc =  HashableQuantumCircuit.from_instructions(self.data, qubits=self.qubits, clbits=self.clbits)
        qc.__class__ = HashableQuantumCircuit
        return qc


def hash_instruction(instruction):
    hashable_params = []
    for param in instruction.operation.params:
        if isinstance(param, numpy.ndarray):
            hashable_params.append(tuple(map(tuple, param)))
        else:
            hashable_params.append(param)
    return hash((instruction.operation.name, tuple(hashable_params), instruction.qubits, instruction.clbits))
