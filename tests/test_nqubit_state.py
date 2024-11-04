import unittest

import numpy as np

from pauliarray import Operator, PauliArray
from pauliarray.state.basis_state_array import BasisStateArray
from pauliarray.state.nqubit_state import NQubitState


class TestQubitState(unittest.TestCase):
    def test_scalar_product(self):
        state1 = NQubitState.from_labels_and_amplitudes(np.array(["00", "11"]), np.array([np.sqrt(0.5), np.sqrt(0.5)]))
        state2 = NQubitState.from_labels_and_amplitudes(np.array(["01", "10"]), np.array([np.sqrt(0.5), np.sqrt(0.5)]))

        state1.scalar_product(state2)

    def test_expected_value(self):
        state = NQubitState.from_labels_and_amplitudes(np.array(["00", "11"]), np.array([np.sqrt(0.5), np.sqrt(0.5)]))
        operator = Operator.from_labels_and_weights(np.array(["XX"]), np.array([1]))

        state.pauli_operator_expectation_value(operator)

    def test_pauli_array_expectation_values(self):
        paulis = PauliArray.from_labels(
            [
                ["XXXX", "IIII", "XXZZ"],
                ["XZYZ", "YXYX", "XZXZ"],
            ]
        )
        state = NQubitState.from_labels_and_amplitudes(
            np.array(["0000", "1111", "1010"]),
            np.sqrt(np.array([1, 1, 1]) / 3),
        )

        expected_expvals = np.zeros(paulis.shape, dtype=complex)
        for idx in np.ndindex(paulis.shape):
            one_pauli = Operator.from_paulis(paulis[idx])
            state_j = state.apply_operator(one_pauli)
            expected_expvals[idx] = state.braket(state_j)

        expectation_values = state.pauli_array_expectation_values(paulis)

        self.assertTrue(np.all(expectation_values == expected_expvals))

    def test_diagonal_pauli_array_expectation_values(self):
        paulis = PauliArray.from_labels(
            [
                ["ZZZZ", "IIII", "ZIZZ"],
                ["IZIZ", "IZZI", "ZIZI"],
            ]
        )
        state = NQubitState.from_labels_and_amplitudes(
            np.array(["0000", "1111", "1010"]),
            np.sqrt(np.array([1, 1, 1]) / 3),
        )

        expected_expvals = np.zeros(paulis.shape, dtype=complex)
        for idx in np.ndindex(paulis.shape):
            one_pauli = Operator.from_paulis(paulis[idx])
            state_j = state.apply_operator(one_pauli)
            expected_expvals[idx] = state.braket(state_j)

        expectation_values = state.pauli_array_expectation_values(paulis)

        self.assertTrue(np.all(expectation_values == expected_expvals))
