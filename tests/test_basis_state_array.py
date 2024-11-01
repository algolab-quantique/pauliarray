import unittest

import numpy as np

from pauliarray import PauliArray
from pauliarray.state.basis_state_array import BasisStateArray


class TestBasisStateArray(unittest.TestCase):
    def test_apply_pauli_array(self):
        paulis = PauliArray.from_labels(["IZXX", "XXYI", "XXZZ"])
        states = BasisStateArray.from_labels(["0000", "0011", "0110"])

        expected_states = BasisStateArray.from_labels(["0011", "1101", "1010"])
        expected_phases = np.array([1, -1j, -1])

        new_states, phases = states.apply_pauli_array(paulis)

        self.assertTrue(np.all(expected_states == new_states))
        self.assertTrue(np.all(expected_phases == phases))

        expected_states = BasisStateArray.from_labels(
            [
                ["0011", "1110", "1100"],
                ["0000", "1101", "1111"],
                ["0101", "1000", "1010"],
            ]
        )
        expected_phases = np.array(
            [
                [1, 1j, 1],
                [1, -1j, 1],
                [-1, -1j, -1],
            ]
        )

        new_states, phases = states[:, None].apply_pauli_array(paulis[None, :])

        self.assertTrue(np.all(expected_states == new_states))
        self.assertTrue(np.all(expected_phases == phases))
