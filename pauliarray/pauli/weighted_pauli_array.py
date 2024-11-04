from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pauliarray.pauli.pauli_array as pa
from pauliarray.utils import label_utils
from pauliarray.utils.array_operations import broadcast_shape, is_broadcastable, is_concatenatable

if TYPE_CHECKING:
    from pauliarray.pauli.operator import Operator
    from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1


class WeightedPauliArray(object):
    def __init__(self, paulis: pa.PauliArray, weights: Union["np.ndarray[np.complex128]", Number]):

        weights = np.atleast_1d(weights)

        if not np.all(weights.shape == paulis.shape):
            shape = broadcast_shape(weights.shape, paulis.shape)

            paulis = pa.broadcast_to(paulis, shape)
            weights = np.broadcast_to(weights, shape)

        self._weights = weights
        self._paulis = paulis

    @property
    def num_qubits(self) -> int:
        return self._paulis.num_qubits

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._weights.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def weights(self) -> "np.ndarray[np.complex128]":
        return self._weights

    @property
    def paulis(self) -> pa.PauliArray:
        return self._paulis

    def __getitem__(self, key):
        new_paulis = self._paulis[key]
        new_weights = self._weights[key]

        return WeightedPauliArray(new_paulis, new_weights)

    def __setitem__(self, key, value: "WeightedPauliArray"):
        if isinstance(value, WeightedPauliArray):
            self._weights[key] = value._weights
            self._paulis[key] = value._paulis
        else:
            raise ValueError()

    def __str__(self):
        return f"WeightedPauliArray: num_qubits = {self.num_qubits}, shape = {str(self.shape)}, ..."

    def __eq__(self, other: "WeightedPauliArray") -> "np.ndarray[np.bool]":
        """
        Checks element-wise if the other WeightedPauliArray is equal.

        Args:
            other (WeightedPauliArray): An other WeightedPauliArray. Must be broadcastable

        Returns:
            "np.ndarray[np.bool]": _description_
        """
        eq_paulis = self.paulis == other.paulis
        eq_weights = np.isclose(self.weights, other.weights)

        return np.logical_and(eq_paulis, eq_weights)

    def _mul(self, other: Union[Number, ArrayLike, "WeightedPauliArray"]) -> "WeightedPauliArray":
        if isinstance(other, Number):
            return self.mul_weights(other)
        elif isinstance(other, WeightedPauliArray):
            return self.compose_weighted_pauli_array(other)

        return NotImplemented

    __mul__ = __rmul__ = _mul

    def copy(self) -> "WeightedPauliArray":
        """
        Returns a copy of the WeightedPauliArray.

        Returns:
            WeightedPauliArray: Copied WeightedPauliArray.
        """
        return WeightedPauliArray(self._paulis.copy(), self._weights.copy())

    def adjoint(self) -> "WeightedPauliArray":
        return WeightedPauliArray(self.paulis, np.conj(self.weights))

    def reshape(self, shape: Tuple[int, ...]) -> "WeightedPauliArray":
        """
        Reshape the WeightedPauliArray

        Args:
            shape (tuple[int]): New shape

        Returns:
            WeightedPauliArray: Reshaped WeightedPauliArray
        """

        # TODO check number of dimensions in shape

        new_weights = self._weights.reshape(shape)
        new_paulis = self._paulis.reshape(shape)

        return WeightedPauliArray(new_paulis, new_weights)

    def flatten(self) -> "WeightedPauliArray":
        """
        Returns a copy of the WeightedPauliArray flattened into one dimension.

        Returns:
            WeightedPauliArray: A flattened copy of the current WeightedPauliArray.
        """

        shape = (np.prod(self.shape, dtype=int),)

        return self.reshape(shape)

    def squeeze(self) -> "WeightedPauliArray":
        """
        Returns a WeightedPauliArray with axes of length one removed.

        Returns:
            WeightedPauliArray: The squeezed WeightedPauliArray.
        """
        new_paulis = self.paulis.squeeze()
        new_weights = self.weights.squeeze()

        return WeightedPauliArray(new_paulis, new_weights)

    def remove(self, index: int) -> "WeightedPauliArray":
        """
        Returns a WeightedPauliArray with removed item at given index.

        Args:
            index (int): Index of element to remove.

        Returns:
            WeightedPauliArray: WeightedPauliArray with removed item at given index.
        """
        new_paulis = self.paulis.remove(index)
        new_weights = self.weights.remove(index)

        return WeightedPauliArray(new_paulis, new_weights)

    def partition(self, parts_flat_idx: List[NDArray[np.int_]]) -> List["WeightedPauliArray"]:
        """
        Returns a list of WeightedPauliArray

        Args:
            parts_flat_idx [List[NDArray[np.int_]]]: List of parts given in linear indices

        Returns:
            List[PauliArray]: Parts
        """

        flat_wpaulis = self.flatten()

        parts = []
        for part_flat_idx in parts_flat_idx:
            parts.append(flat_wpaulis[part_flat_idx])

        return parts

    def partition_with_fct(self, partition_fct: Callable) -> List["WeightedPauliArray"]:
        """
        Returns a list of WeightedPauliArray

        Args:
            partition_fct [Callable]: ...

        Returns:
            List[PauliArray]: Parts
        """

        parts_flat_idx = partition_fct(self.paulis)

        return self.partition(parts_flat_idx)

    def extract(self, condition: Union[NDArray, list]) -> "WeightedPauliArray":
        """
        Return the Pauli strings from the WeightedPauliArray object that satisfy some condition.

        Args:
          condition (Union[NDArray, list]): An array whose nonzero or True entries indicate the Pauli strings of WeightedPauliArray to extract.

        Returns:
            WeightedPauliArray: A new WeightedPauliArray object containing the extracted Pauli strings.

        Raises:
            ValueError: If the shape of the condition array is not equal to shape of the WeightedPauliArray.
        """
        if isinstance(condition, list):
            condition = np.array(condition, dtype=bool)

        if condition.shape != self.shape:
            raise ValueError("The condition array must have the same shape as the weighted Paulis.")

        new_weights = self.weights[condition]
        new_paulis = self.paulis[condition]

        if len(new_weights) == 0:
            return WeightedPauliArray.empty(self.num_qubits)

        return WeightedPauliArray(new_paulis, new_weights)

    def take_qubits(self, indices: Union["np.ndarray[np.int]", range, int]) -> "WeightedPauliArray":
        if isinstance(indices, int):
            indices = np.array([indices], dtype=int)

        new_weights = self.weights.copy()
        new_paulis = self.paulis.take_qubits(indices)

        return WeightedPauliArray(new_paulis, new_weights)

    def compress_qubits(self, condition: "np.ndarray[np.bool]") -> "WeightedPauliArray":
        new_weights = self.weights.copy()
        new_paulis = self.paulis.compress_qubits(condition)

        return WeightedPauliArray(new_paulis, new_weights)

    def compose(self, other: Any) -> Any:

        if isinstance(other, WeightedPauliArray):
            return self.compose_weighted_pauli_array(other)

        return NotImplemented

    def compose_weighted_pauli_array(self, other: "WeightedPauliArray") -> "WeightedPauliArray":
        new_paulis, phases = self._paulis.compose_pauli_array(other.paulis)
        new_weights = self._weights * other.weights * phases

        return WeightedPauliArray(new_paulis, new_weights)

    def mul_weights(self, other: Union[Number, NDArray]) -> "WeightedPauliArray":
        new_weights = self.weights * other
        new_paulis = pa.broadcast_to(self.paulis, new_weights.shape)

        return WeightedPauliArray(new_paulis, new_weights)

    def tensor(self, other: Any) -> Any:

        if isinstance(other, WeightedPauliArray):
            return self.tensor_weighted_pauli_array(other)

        return NotImplemented

    def tensor_weighted_pauli_array(self, other: "WeightedPauliArray") -> "WeightedPauliArray":
        new_paulis = self.paulis.tensor_pauli_array(other.paulis)
        new_weights = self.weights * other.weights

        return WeightedPauliArray(new_paulis, new_weights)

    def add_weighted_pauli_array(self, other: "WeightedPauliArray") -> "OperatorArrayType1":

        from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1

        assert self.num_qubits == other.num_qubits
        assert is_broadcastable(self.shape, other.shape)

        new_weights = np.stack((self.weights, other.weights), axis=-1)
        new_paulis = self.paulis.add_pauli_array(other.paulis)

        new_wpaulis = WeightedPauliArray(new_paulis, new_weights)

        return OperatorArrayType1.from_weighted_pauli_array(new_wpaulis, -1)

    def commute_with(self, other: "WeightedPauliArray") -> "np.ndarray[np.bool]":
        return self.paulis.commute_with(other.paulis)

    def bitwise_commute_with(self, other: "WeightedPauliArray") -> "np.ndarray[np.bool]":
        return self.paulis.bitwise_commute_with(other.paulis)

    def inspect(self) -> str:
        if self.ndim == 0:
            return "Empty PauliArray"

        if self.ndim == 1:
            label_table = label_utils.weighted_table_1d(self.paulis.to_labels(), self.weights)
            return f"PauliArray\n{label_table}"

        if self.ndim == 2:
            label_table = label_utils.weighted_table_2d(self.paulis.to_labels(), self.weights)
            return f"PauliArray\n{label_table}"

        label_table = label_utils.weighted_table_nd(self.paulis.to_labels(), self.weights)
        return f"PauliArray\n{label_table}"

    def x(self, qubits: Union[int, List[int]], inplace: bool = True) -> "WeightedPauliArray":
        """
        Apply X transformations on qubits of WeightedPauliStrings. This leaves the PauliStrings unchanged but produce
        phase factors -1 when operators are Y or Z.

        Args:
            qubits (int or list[int]): The qubits on which to apply the X.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            self_copy (WeightedPauliArray): A modified copy of self, only if inplace=True
        """

        if not inplace:
            return self.copy().x(qubits)

        _, factors = self.paulis.x(qubits, inplace=True)
        self._weights *= factors
        return self

    def h(self, qubits: Union[int, List[int]], inplace: bool = True) -> "WeightedPauliArray":
        """
        Apply H transformations on qubits of WeightedPauliStrings. This exchanges X for Z and vice-versa and Y into -Y.

        Args:
            qubits (int or list[int]): The qubits on which to apply the H.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            self_copy (WeightedPauliArray): A modified copy of self, only if inplace=True
        """
        if not inplace:
            return self.copy().h(qubits)

        _, factors = self.paulis.h(qubits, inplace=True)
        self._weights *= factors
        return self

    def s(self, qubits: Union[int, List[int]], inplace: bool = True) -> "WeightedPauliArray":
        """
        Apply S transformations on qubits of WeightedPauliStrings. This exchanges X for Y and vice-versa with respective factors.

        Args:
            qubits (int or list[int]): The qubits on which to apply the S.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            self_copy (WeightedPauliArray): A modified copy of self, only if inplace=True
        """
        if not inplace:
            return self.copy().s(qubits)

        _, factors = self.paulis.s(qubits, inplace=True)
        self._weights *= factors
        return self

    def cx(
        self,
        control_qubits: Union[int, List[int]],
        target_qubits: Union[int, List[int]],
        inplace: bool = True,
    ) -> "WeightedPauliArray":
        """
        Apply CX transformations on qubits of WeightedPauliStrings. The order of the CX is set by the order of the qubits.

        Args:
            control_qubits (int or list[int]): The qubits which controls the CZ.
            target_qubits (int or list[int]): The qubits target by CZ.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            self_copy (WeightedPauliArray): A modified copy of self, only if inplace=True
        """
        if not inplace:
            return self.copy().cx(control_qubits, target_qubits)

        _, factors = self.paulis.cx(control_qubits, target_qubits, inplace=True)
        self._weights *= factors
        return self

    def cz(
        self,
        control_qubits: Union[int, List[int]],
        target_qubits: Union[int, List[int]],
        inplace: bool = True,
    ) -> "WeightedPauliArray":
        """
        Apply CZ transformations on qubits of WeightedPauliStrings. The order of the CZ is set by the order of the qubits.

        Args:
            control_qubits (int or list[int]): The qubits which controls the CZ.
            target_qubits (int or list[int]): The qubits target by CZ.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            self_copy (WeightedPauliArray): A modified copy of self, only if inplace=True
        """
        if not inplace:
            return self.copy().cz(control_qubits, target_qubits)

        _, factors = self.paulis.cz(control_qubits, target_qubits, inplace=True)
        self._weights *= factors
        return self

    def clifford_conjugate(self, clifford: "Operator", inplace: bool = True) -> "WeightedPauliArray":
        """
        Performs a Clifford transformation.

        Args:
            clifford (Operator) : Must represent a Clifford transformation with the correct number of qubits.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            WeightedPauliArray: The transformed WeightedPauliArray
        """

        new_paulis, factors = clifford.clifford_conjugate_pauli_array_old(self.paulis)
        new_weights = self.weights * factors
        if inplace:
            self._paulis = new_paulis
            self._weights = new_weights

            return self

        return WeightedPauliArray(new_paulis, new_weights)

    def expectation_values_from_paulis(
        self, paulis_expectation_values: NDArray[np.float64]
    ) -> "np.ndarray[np.complex128]":
        """
        Returns the WeightedPauliArray expectation value given the expectation values of the Paulis.

        Args:
            paulis_expectation_values (NDArray[float]): _description_

        Returns:
            NDArray: _description_
        """

        assert np.all(paulis_expectation_values.shape == self.shape)

        return self.weights * paulis_expectation_values

    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float64]) -> "np.ndarray[np.complex128]":
        """
        Returns the WeightedPauliArray covariances given the covariances of the Paulis.

        Args:
            paulis_covariances (NDArray[float]): _description_

        Returns:
            NDArray: _description_
        """
        assert np.all(paulis_covariances.shape == (self.shape + self.shape))

        flat_weights = self.weights.flatten()
        flat_paulis_covariances = paulis_covariances.reshape((self.size, self.size))

        flat_wpaulis_covariances = flat_weights[:, None] * flat_weights[None, :].conj() * flat_paulis_covariances

        return flat_wpaulis_covariances.reshape((self.shape + self.shape))

    def update_weights(self, new_weights):
        assert np.all(self.weights.shape == new_weights.shape)

        self._weights = new_weights.copy()

    def update_weights_from_other(self, other: "WeightedPauliArray"):
        assert np.all(self.paulis == other.paulis)

        self.update_weights(other.weights)

    def is_diagonal(self) -> "np.ndarray[np.bool]":
        """
        Checks if the Pauli strings are diagonal i.e. if all Pauli strings are I or Z.

        Returns:
            NDArray[bool]: True if the Pauli string is diagonal, False otherwise.
        """
        return self._paulis.is_diagonal()

    def to_matrices(self) -> NDArray:
        """
        Returns the WeightedPauliArray as a numpy matrix.

        Returns:
            matrices (NDArray): An ndarray of shape self.shape + (n**2, n**2).
        """

        return self.weights[..., None, None] * self.paulis.to_matrices()

    @classmethod
    def new(cls, shape: Tuple[int, ...], num_qubits: int) -> "WeightedPauliArray":
        weights = np.zeros(shape, dtype=np.complex128)
        paulis = pa.PauliArray.identities(shape, num_qubits)

        return WeightedPauliArray(paulis, weights)

    @classmethod
    def empty(cls, num_qubits: int) -> "WeightedPauliArray":
        """
        Returns an empty WeightedPauliArray with the number of qubits already set.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            WeightedPauliArray: An empty WeightedPauliArray
        """
        weights = np.zeros((), dtype=complex)
        paulis = pa.PauliArray.identities((), num_qubits)

        return WeightedPauliArray(paulis, weights)

    @classmethod
    def random(cls, shape: Tuple[int, ...], num_qubits: int) -> "WeightedPauliArray":
        """
        Creates a WeightedPauliArray of a given shape and number of qubits filled with random Pauli strings and weights.

        Args:
            shape (Tuple[int, ...]): Shape of new PauliArray.
            num_qubits (int): Number of qubits of new PauliArray.

        Returns:
            new_PauliArray (PauliArray): The PauliArray created.
        """
        random_paulis = pa.PauliArray.random(shape, num_qubits)
        random_weights = np.random.rand(*shape)

        return WeightedPauliArray(random_paulis, random_weights)

    @classmethod
    def from_labels_and_weights(cls, labels, weights) -> "WeightedPauliArray":
        paulis = pa.PauliArray.from_labels(labels)

        return WeightedPauliArray(paulis, weights)

    @classmethod
    def from_z_strings_and_x_strings_and_weights(
        cls,
        z_strings: "np.ndarray[np.bool]",
        x_strings: "np.ndarray[np.bool]",
        weights: "np.ndarray[np.complex128]",
    ) -> "WeightedPauliArray":

        paulis = pa.PauliArray(z_strings, x_strings)

        return WeightedPauliArray(paulis, weights)

    @classmethod
    def from_paulis(cls, paulis: pa.PauliArray) -> "WeightedPauliArray":
        weights = np.ones(paulis.shape, dtype=complex)

        return WeightedPauliArray(paulis.copy(), weights)

    def to_npz(self, filename):
        with open(filename, "wb") as f:
            np.save(f, self.paulis.zx_strings)
            np.save(f, self.weights)

    @classmethod
    def from_npz(cls, filename) -> "WeightedPauliArray":
        with open(filename, "rb") as f:
            zx_strings = np.load(f)
            weights = np.load(f)

        return WeightedPauliArray(pa.PauliArray.from_zx_strings(zx_strings), weights)


def broadcast_to(wpaulis: WeightedPauliArray, shape: Tuple[int, ...]) -> "WeightedPauliArray":
    """
    Returns the given WeightedPauliArray broadcasted to a given shape.

    Args:
        paulis (WeightedPauliArray): WeightedPauliArray to broadcast.
        shape (Tuple[int, ...]): Shape to broadcast to.

    Returns:
        new_pauli_array (WeightedPauliArray): The WeightedPauliArray with a new shape.
    """

    new_paulis = pa.broadcast_to(wpaulis.paulis, shape)
    new_weights = np.broadcast_to(wpaulis.weights, shape)

    return WeightedPauliArray(new_paulis, new_weights)


def expand_dims(wpaulis: WeightedPauliArray, axis=Union[int, Tuple[int, ...]]) -> "WeightedPauliArray":
    """
    Expands the shape of a WeightedPauliArray.

    Inserts a new axis that will appear at the axis position in the expanded array shape.

    Args:
        paulis (WeightedPauliArray): The WeightedPauliArray to expand.
        axis (Union[int, Tuple[int, ...]]): The axis upon which expand the WeightedPauliArray.

    Returns:
        expanded_pauli_array (WeightedPauliArray) : The expanded WeightedPauliArray.
    """

    new_paulis = pa.expand_dims(wpaulis.paulis, axis)
    new_weights = np.expand_dims(wpaulis.weights, axis)

    return WeightedPauliArray(new_paulis, new_weights)


def commutator(wpaulis_1: WeightedPauliArray, wpaulis_2: WeightedPauliArray) -> WeightedPauliArray:
    """
    Returns the commutator of the two WeightedPauliArray parameters.

    Args:
        wpaulis_1 (WeightedPauliArray): WeightedPauliArray to calculate commmutator with.
        wpaulis_2 (WeightedPauliArray): Other WeightedPauliArray to calculate commmutator with.

    Returns:
        commutator_pauli_array (WeightedPauliArray): WeightedPauliArray containing the commutators.
    """

    assert is_broadcastable(wpaulis_1.shape, wpaulis_2.shape)

    commutators = wpaulis_1.compose_weighted_pauli_array(wpaulis_2)
    do_commute = wpaulis_1.commute_with(wpaulis_2)

    commutators.paulis.x_strings[do_commute] = 0
    commutators.paulis.z_strings[do_commute] = 0

    commutators._weights *= ~do_commute * 2

    return commutators


def anticommutator(wpaulis_1: WeightedPauliArray, wpaulis_2: WeightedPauliArray) -> WeightedPauliArray:
    assert is_broadcastable(wpaulis_1.shape, wpaulis_2.shape)

    anticommutators = wpaulis_1.compose_weighted_pauli_array(wpaulis_2)
    do_commute = wpaulis_1.commute_with(wpaulis_2)

    anticommutators.paulis.x_strings[~do_commute] = 0
    anticommutators.paulis.z_strings[~do_commute] = 0

    anticommutators._weights *= do_commute * 2

    return anticommutators


def concatenate(wpauli_arrays: Tuple[WeightedPauliArray, ...], axis: int) -> WeightedPauliArray:
    """
    Concatenated multiple WeightedPauliArrays.

    Args:
        paulis (List[PauliArray]): WeightedPauliArrays to concatenate.
        axis (int): The axis along which the arrays will be joined.

    Returns:
        WeightedPauliArray: The concatenated WeightedPauliArrays.
    """

    assert is_concatenatable(wpauli_arrays, axis)

    weights_list = tuple(wpaulis.weights for wpaulis in wpauli_arrays)
    paulis_list = tuple(wpaulis.paulis for wpaulis in wpauli_arrays)

    new_weights = np.concatenate(weights_list, axis)
    new_paulis = pa.concatenate(paulis_list, axis)

    return WeightedPauliArray(new_paulis, new_weights)


def swapaxes(wpaulis: WeightedPauliArray, axis1: int, axis2: int):
    """
    Swap axes of a WeightedPauliArray

    Args:
        paulis (WeightedPauliArray): The WeightedPauliArray
        axis1 (int): Original axis position
        axis2 (int): Target axis position

    Returns:
        WeightedPauliArray: The WeightedPauliArrays with axes swaped.
    """

    assert axis1 < wpaulis.ndim
    assert axis2 < wpaulis.ndim

    new_weights = np.swapaxes(wpaulis.weights, axis1, axis2)
    new_paulis = pa.swapaxes(wpaulis.paulis, axis1, axis2)

    return WeightedPauliArray(new_paulis, new_weights)


def moveaxis(wpaulis: WeightedPauliArray, source: int, destination: int):
    """
    Move an axis of a WeightedPauliArray

    Args:
        paulis (WeightedPauliArray): The WeightedPauliArray
        axis1 (int): Original axis position
        axis2 (int): Target axis position

    Returns:
        WeightedPauliArray: The WeightedPauliArrays with axis moved.
    """

    assert source < wpaulis.ndim
    assert destination < wpaulis.ndim

    new_weights = np.moveaxis(wpaulis.weights, source, destination)
    new_paulis = pa.moveaxis(wpaulis.paulis, source, destination)

    return WeightedPauliArray(new_paulis, new_weights)
