import unittest

import numpy as np

from pauliarray.partition.commutating_paulis.exclusive_fct import (
    partition_bitwise_commutating,
    partition_general_commutating,
    partition_same_x,
)
from pauliarray.pauli.operator import Operator
from pauliarray.pauli.pauli_array import PauliArray
from pauliarray.pauli.weighted_pauli_array import WeightedPauliArray


class TestPartition(unittest.TestCase):
    def test_partition_pauli_bitwise_commutating(self):

        paulis = PauliArray.random((10, 10), 5)

        parts_flat_idx = partition_bitwise_commutating(paulis)

        parts = paulis.partition(parts_flat_idx)

        for part in parts:
            self.assertTrue(np.all(part[:, None].bitwise_commute_with(part[None, :])))

    def test_partition_pauli_general_commutating(self):

        paulis = PauliArray.random((10, 10), 5)

        parts_flat_idx = partition_general_commutating(paulis)

        parts = paulis.partition(parts_flat_idx)

        for part in parts:
            self.assertTrue(np.all(part[:, None].commute_with(part[None, :])))

    def test_partition_same_x(self):

        paulis = PauliArray.random((10, 10), 5)

        parts_flat_idx = partition_same_x(paulis)

        parts = paulis.partition(parts_flat_idx)

        for part in parts:
            self.assertTrue(np.all(part[:, None].commute_with(part[None, :])))

    def test_partition_weighted_pauli_general_commutating(self):

        wpaulis = WeightedPauliArray.random((10, 10), 5)

        parts_flat_idx = partition_general_commutating(wpaulis)

        parts = wpaulis.partition(parts_flat_idx)

        for part in parts:
            self.assertTrue(np.all(part[:, None].commute_with(part[None, :])))

    def test_partition_operator_general_commutating(self):

        operator = Operator.random(100, 5)

        parts_flat_idx = partition_general_commutating(operator)

        parts = operator.partition(parts_flat_idx)

        for part in parts:
            self.assertTrue(np.all(part.paulis[:, None].commute_with(part.paulis[None, :])))
