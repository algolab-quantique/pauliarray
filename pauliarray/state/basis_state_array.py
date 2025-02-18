from typing import Tuple, Union

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops
from pauliarray.utils.array_operations import broadcast_shape, is_broadcastable, is_concatenatable
from pauliarray.utils.label_utils import table_1d, table_2d, table_nd

BIT_LABELS = "01"


class BasisStateArray(object):
    """
    Defines an array of computationnal basis states.
    """

    def __init__(self, bit_strings: "np.ndarray[np.bool]"):

        bit_strings = np.atleast_2d(bit_strings)

        assert bit_strings.dtype == np.dtype(np.bool_)

        self._bit_strings = bit_strings

    @property
    def bit_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the bits.

        Returns:
            "np.ndarray[np.bool]": The bits.
        """
        return self._bit_strings

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self.bit_strings.shape[-1]

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the object.

        Returns:
            Tuple[int, ...]: The shape of the object.
        """
        return self.bit_strings.shape[:-1]

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
        """
        Transform each basis state by applying a PauliArray element wise on it.

        Returns:
            BasisStateArray: The resulting BasisState.
            "np.ndarray[np.complex]": The resulting phases.
        """

        assert self.num_qubits == paulis.num_qubits
        assert is_broadcastable(self.shape, paulis.shape)

        new_bit_strings = bitops.add(self.bit_strings, paulis.x_strings)

        commutation_phase_power = 2 * bitops.dot(paulis.z_strings, new_bit_strings)
        phase_power = bitops.dot(paulis.z_strings, paulis.x_strings)

        phase_power = np.mod(commutation_phase_power + phase_power, 4)

        phases = np.choose(phase_power, [1, -1j, -1, 1j])
        new_basis_states = BasisStateArray(new_bit_strings)

        return new_basis_states, phases

    def scalar_product(self, other: "BasisStateArray") -> "np.ndarray[np.bool]":
        """
        Performs a scale product between the basis states in both BasisStateArray element wise.
        Since all computationnal basis states are orthonormal, the result is 1 if the basis states are equal and 0 otherwise.

        Args:
            other (BasisStateArray): An other broadcastable BasisStateArray

        Returns:
            np.ndarray[np.bool]: The resulting scalar product.
        """

        assert self.num_qubits == other.num_qubits
        assert is_broadcastable(self.shape, other.shape)

        return self == other

    def diagonal_pauli_array_eigenvalues_values(self, diag_paulis: pa.PauliArray) -> "np.ndarray[np.complex128]":
        """
        Computes the eigenvalues of diagonal pauli strings for the basis states. This is done element wise so the PauliArray has to be broadcastable with the BasisStateArray.

        Args:
            diag_paulis (pa.PauliArray): A broadcastable PauliArray of diagonal Pauli strings.

        Returns:
            np.ndarray[np.complex128]: The eigenvalues.
        """
        assert np.all(diag_paulis.is_diagonal())

        assert self.num_qubits == diag_paulis.num_qubits
        assert is_broadcastable(self.shape, diag_paulis.shape)

        eigenvalues = 1 - 2 * np.mod(bitops.dot(diag_paulis.z_strings, self.bit_strings), 2)

        return eigenvalues

    def pauli_array_expectation_value(self, paulis: pa.PauliArray) -> "np.ndarray[np.complex128]":
        """
        Computes the expectation values of pauli strings for the basis states. This is done element wise so the PauliArray has to be broadcastable with the BasisStateArray.

        Args:
            paulis (pa.PauliArray): A broadcastable PauliArray.

        Returns:
            np.ndarray[np.complex128]: The expectation values.
        """
        mod_self, phase = self.apply_pauli_array(paulis)

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
            label_table = table_1d(self.to_labels())
            return f"BasisStateArray\n{label_table}"

        if self.ndim == 2:
            label_table = table_2d(self.to_labels())
            return f"BasisStateArray\n{label_table}"

        label_table = table_nd(self.to_labels())
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

    @classmethod
    def from_labels(cls, labels: Union[list[str], "np.ndarray[np.str]"]) -> "BasisStateArray":
        """
        Constructs a BasisStateArray from labels.

        Args:
            labels (Union[list[str], "np.ndarray[np.str]"]): The list of labels.

        Returns:
            BasisStateArray: The basis states.
        """
        if type(labels) not in (list, np.ndarray):
            labels = [labels]

        labels = np.atleast_1d(np.array(labels, dtype=str))

        num_qubits = len(labels[(0,) * labels.ndim])
        shape = labels.shape

        bit_strings = np.zeros(shape + (num_qubits,), dtype=bool)
        for idx in np.ndindex(*shape):
            label = labels[idx]
            bit_strings[idx] = np.array([s == "1" for s in reversed(label)])

        return BasisStateArray(bit_strings)

    @classmethod
    def complete_basis(cls, num_qubits: int) -> "BasisStateArray":
        """
        Constructs a BasisStateArray containing all the basis states for a given number of qubits.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            BasisStateArray: The complete basis state basis.
        """
        bin_power = 2 ** np.arange(num_qubits, dtype=np.uintc)
        bit_strings = ((np.arange(2 ** (num_qubits), dtype=np.uintc)[:, None] & bin_power[None, :]) > 0).reshape(
            (2**num_qubits, num_qubits)
        )

        return BasisStateArray(bit_strings)

    @classmethod
    def integer_subbasis(cls, num_qubits: int, integers: "np.array[np.int64]") -> "BasisStateArray":
        """
        Converts an array of intergers to basis state using their binary representation.

        Args:
            num_qubits (int): The number of qubits. Only the [num_qubits] first bits of the integer should be non zeros.
            integers (np.array[np.int64]): The integers

        Returns:
            BasisStateArray: The basis states.
        """
        assert np.max(integers) < 2**num_qubits

        bin_power = 2 ** np.arange(num_qubits, dtype=np.uintc)
        bit_strings = ((integers[:, None] & bin_power[None, :]) > 0).reshape((len(integers), num_qubits))

        return BasisStateArray(bit_strings)


def unique(
    states: BasisStateArray,
    axis=None,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Union[BasisStateArray, Tuple[BasisStateArray, "np.ndarray[np.int64]"]]:
    """
    Finds unique elements in a BasisStateArray.
    Directly uses numpy.unique and has the same interface.

    Args:
        states (BasisStateArray): The BasisStateArray.

        axis (Optional[int], optional):  The axis to operate on. If None, the BasisStateArray will be flattened.
            If an integer, the subarrays indexed by the given axis will be flattened and treated as the elements
            of a 1-D array with the dimension of the given axis. Object arrays or structured arrays that contain
            objects are not supported if the axis kwarg is used. Defaults to None.

        return_index (bool, optional): If True, also return the indices of BasisStateArray (along the specified axis, if provided, or in the flattened array) that result in the unique array. Defaults to False.

        return_inverse (bool, optional): If True, also return the indices of the unique array
            (for the specified axis, if provided) that can be used to reconstruct array. Defaults to False.

        return_counts (bool, optional): If True, also return the number of times each unique item appears in array.
            Defaults to False.

    Returns:
        BasisStateArray: The unique Pauli strings (or BasisStateArray along an axis) in a BasisStateArray
        NDArray, optional: Index to get unique from the orginal BasisStateArray
        NDArray, optional: Innverse to reconstrut the original BasisStateArray from unique
        NDArray, optional: The number of each unique in the original BasisStateArray
    """

    if axis is None:
        states = states.flatten()
        axis = 0
    elif axis >= states.ndim:
        raise ValueError("")
    else:
        axis = axis % states.ndim

    out = np.unique(
        states.bit_strings,
        axis=axis,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if return_index or return_inverse or return_counts:
        out = list(out)
        unique_bit_strings = out[0]
        out[0] = BasisStateArray(unique_bit_strings)
    else:
        unique_bit_strings = out
        out = BasisStateArray(unique_bit_strings)

    return out


def fast_flat_unique(
    states: BasisStateArray,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Union[BasisStateArray, Tuple[BasisStateArray, "np.ndarray[np.int64]"]]:
    """
    Faster version of unique for BasisStateArray. Only works with flat BasisStateArray.
    Directly uses numpy.unique.

    Args:
        paulis (BasisStateArray): The BasisStateArray. Must be flat.

        return_index (bool, optional): If True, also return the indices of BasisStateArray (along the specified axis, if provided, or in the flattened array) that result in the unique array. Defaults to False.

        return_inverse (bool, optional): If True, also return the indices of the unique array
            (for the specified axis, if provided) that can be used to reconstruct array. Defaults to False.

        return_counts (bool, optional): If True, also return the number of times each unique item appears in array.
            Defaults to False.

    Returns:
        BasisStateArray: The unique Pauli strings in a BasisStateArray
        NDArray, optional: Index to get unique from the orginal BasisStateArray
        NDArray, optional: Innverse to reconstrut the original BasisStateArray from unique
        NDArray, optional: The number of each unique in the original BasisStateArray
    """

    assert states.ndim == 1

    bit_strings = states.bit_strings
    void_type_size = bit_strings.dtype.itemsize * states.num_qubits

    bit_view = np.squeeze(np.ascontiguousarray(bit_strings).view(np.dtype((np.void, void_type_size))), axis=-1)

    _, index, inverse, counts = np.unique(bit_view, return_index=True, return_inverse=True, return_counts=True)

    new_paulis = states[index]

    out = (new_paulis,)
    if return_index:
        out += (index,)
    if return_inverse:
        out += (inverse,)
    if return_counts:
        out += (counts,)

    if len(out) == 1:
        return out[0]

    return out
