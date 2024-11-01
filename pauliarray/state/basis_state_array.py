from typing import Tuple

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops
from pauliarray.utils.array_operations import broadcast_shape, is_broadcastable, is_concatenatable
from pauliarray.utils.label_table import label_table_1d, label_table_2d, label_table_nd

BIT_LABELS = "01"


class BasisStateArray(object):
    def __init__(self, bit_strings: "np.ndarray[np.bool]"):

        bit_strings = np.atleast_2d(bit_strings)

        assert bit_strings.dtype == np.dtype(np.bool_)

        self._bit_strings = bit_strings

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self._bit_strings.shape[-1]

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the object.

        Returns:
            Tuple[int, ...]: The shape of the object.
        """
        return self._bit_strings.shape[:-1]

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions.

        Returns:
            int: The number of dimensions.
        """
        return len(self.shape)

    @property
    def size(self) -> int:
        """
        Returns the total number of elements in the BasisStateArray.

        Returns:
            int: The total number of elements in the BasisStateArray.
        """
        return np.prod(self.shape)

    @property
    def bit_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the bits.

        Returns:
            "np.ndarray[np.bool]": The bits.
        """
        return self._bit_strings

    def __getitem__(self, key):
        # TODO check number of dimensions in key
        new_bit_strings = self._bit_strings[key]

        return BasisStateArray(new_bit_strings)

    def __setitem__(self, key, value: "BasisStateArray"):
        if isinstance(value, BasisStateArray):
            self._bit_strings[key] = value.bit_strings
        else:
            raise TypeError("Value should be of type BasisStateArray.")

    def __str__(self):
        return f"BasisStateArray: num_qubits = {self.num_qubits}, shape = {self.shape}, ..."

    def __eq__(self, other: "BasisStateArray") -> "np.ndarray[np.bool]":
        """
        Checks element-wise if the other BasisStateArray is equal.

        Args:
            other (BasisStateArray): An other BasisStateArray. Must be broadcastable

        Returns:
            "np.ndarray[np.bool]": _description_
        """

        return np.all(self.bit_strings == other.bit_strings, axis=-1)

    def copy(self) -> "BasisStateArray":
        """
        Returns a copy of the BasisStateArray.

        Returns:
            BasisStateArray: Copied BasisStateArray.
        """
        return BasisStateArray(self.bit_strings.copy())

    def reshape(self, shape: Tuple[int, ...]) -> "BasisStateArray":
        """
        Returns a BasisStateArray with a new shape.

        Args:
            shape (tuple[int]): Tuple containing the new shape e.g. for 2x2 matrix: shape = (2,2)

        Returns:
            BasisStateArray: The BasisStateArray object with the new shape.
        """

        # TODO check number of dimensions in shape

        new_shape = shape + (self.num_qubits,)
        new_bit_strings = self._bit_strings.reshape(new_shape)

        return BasisStateArray(new_bit_strings)

    def flatten(self) -> "BasisStateArray":
        """
        Returns a copy of the BasisStateArray flattened into one dimension.

        Returns:
            BasisStateArray: A flattened copy of the current BasisStateArray.
        """
        shape = (np.prod(self.shape),)

        return self.reshape(shape)

    def apply_pauli_array(self, paulis: pa.PauliArray) -> Tuple["BasisStateArray", "np.ndarray[np.complex128]"]:

        assert self.num_qubits == paulis.num_qubits
        assert is_broadcastable(self.shape, paulis.shape)

        new_bit_strings = bitops.add(self.bit_strings, paulis.x_strings)

        commutation_phase_power = bitops.dot(paulis.z_strings, new_bit_strings)
        phase_power = bitops.dot(paulis.z_strings, paulis.x_strings)

        phase_power = np.mod(commutation_phase_power + phase_power, 4)

        phases = np.choose(phase_power, [1, -1j, -1, 1j])
        new_basis_states = BasisStateArray(new_bit_strings)

        return new_basis_states, phases

    def scalar_product(self, other: "BasisStateArray") -> "np.ndarray[np.bool]":

        assert self.num_qubits == other.num_qubits
        assert is_broadcastable(self.shape, other.shape)

        return self == other

    def diagonal_pauli_array_eigenvalues_values(self, paulis: pa.PauliArray) -> "np.ndarray[np.complex128]":
        assert np.all(paulis.is_diagonal())

        assert self.num_qubits == paulis.num_qubits
        assert is_broadcastable(self.shape, paulis.shape)

        eigenvalues = 1 - 2 * np.mod(bitops.dot(paulis.z_strings, self.bit_strings), 2)

        return eigenvalues

    def pauli_array_expectation_value(self, pauli_array: pa.PauliArray) -> "np.ndarray[np.complex128]":
        mod_self, phase = self.apply_pauli_array(pauli_array)

        return phase * mod_self.scalar_product(self)

    def inspect(self):
        """
        Returns an inspection string showing all labels of the BasisStateArray.

        Returns:
            str: The inspection string.
        """
        if self.ndim == 0:
            return "Empty BasisStateArray"

        if self.ndim == 1:
            label_table = label_table_1d(self.to_labels())
            return f"BasisStateArray\n{label_table}"

        if self.ndim == 2:
            label_table = label_table_2d(self.to_labels())
            return f"BasisStateArray\n{label_table}"

        label_table = label_table_nd(self.to_labels())
        return f"BasisStateArray\n{label_table}"

    def to_labels(self):
        """
        Returns the labels of all zx strings.

        Returns:
            "np.ndarray[np.str]": An array containing the labels of all Pauli strings.
        """
        labels = np.zeros(self.shape, dtype=f"U{self.num_qubits}")

        for idx in np.ndindex(*self.shape):
            label = ""
            for bit in reversed(self.bit_strings[idx]):
                label += BIT_LABELS[bit]
            labels[idx] = label

        return labels
