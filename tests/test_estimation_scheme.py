import unittest

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_circuits import diagonalise_with_circuits
from pauliarray.estimation.scheme import EstimationSchemeExclusivePartition
from pauliarray.partition.commutating_paulis.exclusive_fct import partition_general_commutating, partition_same_x


class TestEstimationSchemeExclusivePartition(unittest.TestCase):
    def test_prepare(self):

        paulis = pa.PauliArray.random((3, 5), 6)

        scheme = EstimationSchemeExclusivePartition(paulis, partition_general_commutating, diagonalise_with_circuits)

        for part in scheme.diag_parts:
            print(part.inspect())

        for circuit in scheme.parts_transformation:
            print(circuit)
