from numbers import Number
from typing import Tuple, Union

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.operator_array_type_1 as opa
import pauliarray.pauli.pauli_array as pa
import pauliarray.state.basis_state_array as bsa
from pauliarray.binary import bit_operations as bitops
from pauliarray.utils import label_utils


class NQubitState(object):
    def __init__(self, basis: bsa.BasisStateArray, amplitudes: "np.ndarray[np.complex]"):

        assert basis.ndim == 1
        assert amplitudes.ndim == 1
        assert basis.shape == amplitudes.shape

        self._basis = basis
        self._amplitudes = amplitudes

    @property
    def basis(self) -> bsa.BasisStateArray:
        """
        Returns the basis states.

        Returns:
            bsa.BasisStateArray: Basis states.
        """
        return self._basis

    @property
    def amplitudes(self) -> "np.ndarray[np.complex128]":
        """
        Returns the amplitudes associated with basis states.

        Returns:
            "np.ndarray[np.complex128]": Array of complex number amplitudes.
        """
        return self._amplitudes

    @property
    def bit_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the bit strings constructing the basis states.

        Returns:
            "np.ndarray[np.bool]": Array of bit strings.
        """
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
        """
        Returns the number of terms in state.

        Returns:
            int: The number of terms.
        """
        return self.basis.size

    def copy(self) -> "NQubitState":

        new_amplitudes = self.amplitudes.copy()
        new_basis = self.basis.copy()

        return NQubitState

    def adjoint(self) -> "NQubitState":
        """
        Returns the adjoint of current qubit state.

        Returns:
            "NQubitState": Adjoint qubit state.
        """

        new_amplitudes = np.conj(self._amplitudes)
        new_basis = self.basis.copy()

        return NQubitState(new_basis, new_amplitudes)

    def normalise(self) -> "NQubitState":
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        new_amplitudes = self.amplitudes / norm

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

    def __eq__(self, other: "NQubitState") -> "np.ndarray[np.bool]":
        """
        Checks if the other NQubitState is equal.

        Args:
            other (BasisStateArray): An other BasisStateArray. Must be broadcastable

        Returns:
            "np.ndarray[np.bool]": _description_
        """

        return np.all(self.basis == other.basis, axis=-1) and np.all(np.isclose(self.amplitudes, other.amplitudes))

    def combine_repeated_terms(self) -> "NQubitState":
        """
        Combine repeated basis state in the sum by adding their amplitudes.

        Returns:
            NQubitState: _description_
        """
        new_basis, inverse = bsa.fast_flat_unique(self.basis, return_inverse=True)

        new_amplitudes = np.zeros(new_basis.shape, dtype=self.amplitudes.dtype)
        np.add.at(new_amplitudes, inverse, self.amplitudes)

        return NQubitState(new_basis, new_amplitudes)

    def remove_small_amplitudes(self, threshold: float = 1e-12) -> "NQubitState":
        """
        Remove small amplitudes from the NQubitState.

        Args:
            threshold (float, optional): The threshold below which amplitudes are considered small. Defaults to 1e-14.

        Returns:
            NQubitState: The Operator with small amplitudes removed.
        """

        threshold_mask = np.abs(self.amplitudes) > threshold

        return NQubitState(self.basis[threshold_mask], self.amplitudes[threshold_mask])

    def apply_operator(self, operator: op.Operator):
        """
        Apply an Operator on the NQubitState. O|psi>

        Args:
            pauli_operator (op.Operator): An operator

        Returns:
            NQubitState: The transformed quantum state.
        """
        new_basis, phases = self.basis[:, None].apply_pauli_array(operator.paulis[None, :])
        new_amplitudes = phases * self.amplitudes[:, None] * operator.weights[None, :]

        return NQubitState(new_basis.flatten(), new_amplitudes.flatten())

    def apply_operator_array(self, operator_array: opa.OperatorArrayType1):
        """
        Apply each operator on the state, starting with the last one.

        Args:
            operator_array (opa.OperatorArrayType1): _description_
        """

        assert operator_array.ndim == 1

        new_state = self.copy()

        for i in reversed(range(operator_array.size)):
            transformation = operator_array.get_operator(i)
            new_state = new_state.apply_pauli_operator(transformation).simplify()

        return new_state

    def scalar_product(self, other: "NQubitState"):
        """
        Performs a scalar product with an other NQubitState. <self|other>

        Args:
            other (NQubitState): An other NQubitState

        Returns:
            complex: The value of the scale product
        """
        amplitude_array = np.conj(self.amplitudes[:, None]) * other.amplitudes[None, :]
        matching_state_array = self.basis[:, None].scalar_product(other.basis[None, :])

        return np.sum(amplitude_array[matching_state_array])

    braket = scalar_product

    def operator_expectation_value(self, operator: op.Operator) -> complex:
        """
        Computes the expectation value of an Operator of the (self) quantum state. <self|O|self>

        Args:
            operator (op.Operator): An operator

        Returns:
            complex: The expectation value.
        """
        mod_self = self.apply_operator(operator)
        return self.scalar_product(mod_self)

    def pauli_array_expectation_values(self, paulis: pa.PauliArray) -> "np.ndarray[np.complex]":
        """
        Computes the expectation value for all pauli strings in a PauliArray.

        $$(-i)^{z_{nd}x_{nd}}\bra{\phi_i} Z^{z_{nd}} X^{x_{nd}} \ket{\phi_j}$$

        Args:
            paulis (pa.PauliArray): Pauli strings

        Returns:
            np.ndarray[np.complex]: The expectation values.
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

        y_phases = np.choose(np.mod(bitops.dot(paulis.z_strings, paulis.x_strings), 4), [1, -1j, -1, 1j])

        return expectation_values * y_phases

    def diagonal_pauli_array_expectation_values(self, diag_: pa.PauliArray):
        """
        Computes the expectation value for all diagonal pauli strings in a PauliArray. Specialized function that is more efficient for diagonal pauli strings.

        $$(-i)^{z_{nd}x_{nd}}\bra{\phi_i} Z^{z_{nd}} X^{x_{nd}} \ket{\phi_j}$$

        Args:
            diag_ (pa.PauliArray): Pauli strings

        Returns:
            np.ndarray[np.complex]: The expectation values.
        """

        assert np.all(diag_.is_diagonal())

        i_prod_amplitudes = np.conj(self.amplitudes) * self.amplitudes

        nd_i_shape = diag_.shape + (self.num_terms,)

        nd_i_paulis = pa.broadcast_to(pa.expand_dims(diag_, (diag_.ndim,)), nd_i_shape)

        nd_i_bit_strings = np.broadcast_to(
            np.expand_dims(self.bit_strings, tuple(range(0, diag_.ndim))), nd_i_shape + (self.num_qubits,)
        )
        nd_i_phases = np.choose(np.mod(bitops.dot(nd_i_paulis.z_strings, nd_i_bit_strings), 2), [1, -1])

        expectation_values = np.sum(i_prod_amplitudes * nd_i_phases, axis=-1)

        return expectation_values

    def inspect(self) -> str:
        """
        Generates a string representation of the NQubitState showing amplitudes and basis states.

        Returns:
            str: A string representation of the Operator.
        """
        labels = self.basis.to_labels()
        weights = self.amplitudes

        detail_str = "State\nSum of\n"
        detail_str += label_utils.weighted_table_1d(labels, weights)

        return detail_str

    @classmethod
    def from_statevector(cls, statevector: "np.array[np.complex128]", threshold: float = 1e-12) -> "NQubitState":
        """
        Constructs a NQubitState directly from a state vector.

        Args:
            statevector (np.array[np.complex128]): The 2**n amplitudes of the state.
            threshold (float, optional): A threshold under which the basis state is not included in the description of the quantum state. Defaults to 1e-12.

        Returns:
            NQubitState: The quantum state
        """
        num_qubits = int(np.log2(statevector.size))

        nonzero_idx = np.where(np.abs(statevector) > threshold)[0]

        subbasis = bsa.BasisStateArray.from_integers(num_qubits, nonzero_idx)

        return NQubitState(subbasis, statevector[nonzero_idx])

    @classmethod
    def from_labels_and_amplitudes(
        cls, labels: Union[list[str], "np.ndarray[np.str]"], amplitudes: Union["np.ndarray[np.complex]", Number]
    ) -> "NQubitState":
        """
        Constructs a NQubitState from basis state labels and amplitudes.

        Args:
            labels (_type_): _description_
            amplitudes (_type_): _description_

        Returns:
            NQubitState: _description_
        """
        basis_states = bsa.BasisStateArray.from_labels(labels)

        return cls(basis_states, amplitudes)
