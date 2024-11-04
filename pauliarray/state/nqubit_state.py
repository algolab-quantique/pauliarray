from typing import Tuple

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
import pauliarray.state.basis_state_array as bsa
from pauliarray.binary import bit_operations as bitops


class NQubitState(object):
    def __init__(self, basis: bsa.BasisStateArray, amplitudes: "np.ndarray[np.complex]"):

        assert basis.ndim == 1
        assert amplitudes.ndim == 1
        assert basis.shape == amplitudes.shape

        self._basis = basis
        self._amplitudes = amplitudes

    @property
    def basis(self) -> bsa.BasisStateArray:
        return self._basis

    @property
    def amplitudes(self) -> "np.ndarray[np.complex128]":
        return self._amplitudes

    @property
    def bit_strings(self) -> "np.ndarray[np.bool]":
        return self.basis.bit_strings

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self.basis.bit_strings.shape[-1]

    @property
    def num_terms(self) -> int:
        return self.basis.size

    def adjoint(self) -> "NQubitState":

        new_amplitudes = np.conj(self._amplitudes)
        new_basis = self.basis.copy()

        return NQubitState(new_basis, new_amplitudes)

    def __add__(self, other: "NQubitState") -> "NQubitState":
        """
        Adds another NQubitState or a scalar to this NQubitState.

        Args:
            other (NQubitState): Another NQubitState.

        Returns:
            NQubitState: The resulting NQubitState.
        """

        return NotImplemented

    def combine_repeated_terms(self) -> "NQubitState":
        new_basis, inverse = bsa.fast_flat_unique(self.basis, return_inverse=True)

        new_amplitudes = np.zeros(new_basis.shape, dtype=self.amplitudes.dtype)
        np.add.at(new_amplitudes, inverse, self.amplitudes)

        return NQubitState(new_basis, new_amplitudes)

    def remove_small_amplitudes(self, threshold: float = 1e-12) -> "NQubitState":
        threshold_mask = np.abs(self.amplitudes) > threshold

        return NQubitState(self.basis[threshold_mask], self.amplitudes[threshold_mask])

    def apply_operator(self, pauli_operator: op.Operator):
        new_basis, phases = self.basis[:, None].apply_pauli_array(pauli_operator.paulis[None, :])
        new_amplitudes = phases * self.amplitudes[:, None] * pauli_operator.weights[None, :]

        return NQubitState(new_basis.flatten(), new_amplitudes.flatten())

    def scalar_product(self, other: "NQubitState"):
        amplitude_array = np.conj(self.amplitudes[:, None]) * other.amplitudes[None, :]
        matching_state_array = self.basis[:, None].scalar_product(other.basis[None, :])

        return np.sum(amplitude_array[matching_state_array])

    braket = scalar_product

    def pauli_operator_expectation_value(self, operator: op.Operator):
        mod_self = self.apply_operator(operator)
        return self.scalar_product(mod_self)

    def pauli_array_expectation_values(self, paulis: pa.PauliArray):
        """
        \bra{\phi_i} Z^{z_{nd}} X^{x_{nd}} \ket{\phi_j}

        Args:
            paulis (pa.PauliArray): _description_

        Returns:
            _type_: _description_
        """
        ij_prod_amplitudes = np.conj(self.amplitudes[:, None]) * self.amplitudes[None, :]
        ij_bit_strings = np.logical_xor(self.bit_strings[:, None, :], self.bit_strings[None, :, :])

        nd_i_shape = paulis.shape + (self.num_terms,)

        nd_i_paulis = pa.broadcast_to(pa.expand_dims(paulis, (paulis.ndim,)), nd_i_shape)

        nd_i_bit_strings = np.broadcast_to(
            np.expand_dims(self.bit_strings, tuple(range(0, paulis.ndim))), nd_i_shape + (self.num_qubits,)
        )
        nd_i_phases = np.choose(np.mod(bitops.dot(nd_i_paulis.z_strings, nd_i_bit_strings), 2), [1, -1])

        nd_ij_shape = paulis.shape + (self.num_terms, self.num_terms)

        nd_ij_paulis = pa.broadcast_to(pa.expand_dims(paulis, (paulis.ndim, paulis.ndim + 1)), nd_ij_shape)
        nd_ij_bit_strings = np.broadcast_to(
            np.expand_dims(ij_bit_strings, tuple(range(0, paulis.ndim))), nd_ij_shape + (self.num_qubits,)
        )

        nd_ij_prod_amplitudes = np.broadcast_to(
            np.expand_dims(ij_prod_amplitudes, tuple(range(0, paulis.ndim))), nd_ij_shape
        )

        nd_ij_matching_x = np.all(nd_ij_bit_strings == nd_ij_paulis.x_strings, axis=-1)

        expectation_values = np.sum(nd_ij_prod_amplitudes * nd_ij_matching_x * nd_i_phases[..., None], axis=(-1, -2))

        y_phases = np.choose(np.mod(bitops.dot(paulis.z_strings, paulis.x_strings), 4), [1, 1j, -1, -1j])

        return expectation_values * y_phases

    def diagonal_pauli_array_expectation_values(self, paulis: pa.PauliArray):

        assert np.all(paulis.is_diagonal())

        i_prod_amplitudes = np.conj(self.amplitudes) * self.amplitudes

        nd_i_shape = paulis.shape + (self.num_terms,)

        nd_i_paulis = pa.broadcast_to(pa.expand_dims(paulis, (paulis.ndim,)), nd_i_shape)

        nd_i_bit_strings = np.broadcast_to(
            np.expand_dims(self.bit_strings, tuple(range(0, paulis.ndim))), nd_i_shape + (self.num_qubits,)
        )
        nd_i_phases = np.choose(np.mod(bitops.dot(nd_i_paulis.z_strings, nd_i_bit_strings), 2), [1, -1])

        expectation_values = i_prod_amplitudes * nd_i_phases

        return expectation_values

    @classmethod
    def from_statevector(cls, statevector: "np.array[np.complex128]") -> "NQubitState":
        num_qubits = int(np.log2(statevector.size))

        all_basis = bsa.BasisStateArray.complete_basis(num_qubits)

        return NQubitState(all_basis, statevector).remove_small_amplitudes()

    @classmethod
    def from_labels_and_amplitudes(cls, labels, amplitudes) -> "NQubitState":
        basis_states = bsa.BasisStateArray.from_labels(labels)

        return cls(basis_states, amplitudes)
