import re
from numbers import Number
from typing import TYPE_CHECKING, Any, List, Literal, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pauliarray.pauli.pauli_array as pa
from pauliarray.utils.array_operations import broadcast_shape, is_broadcastable, is_concatenatable

if TYPE_CHECKING:
    from pauliarray.pauli.operator import Operator
    from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1


class PhasedPauliArray(object):
    def __init__(self, paulis: pa.PauliArray, phases: NDArray[np.uint]):

        phases = np.atleast_1d(phases)

        if not np.all(phases.shape == paulis.shape):
            shape = broadcast_shape(phases.shape, paulis.shape)

            paulis = pa.broadcast_to(paulis, shape)
            phases = np.broadcast_to(phases, shape)

        self._phases = phases
        self._paulis = paulis

    @property
    def num_qubits(self) -> int:
        return self._paulis.num_qubits

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._phases.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def phases(self) -> "np.ndarray[np.complex]":
        return self._phases

    @property
    def paulis(self) -> pa.PauliArray:
        return self._paulis

    def __getitem__(self, key):
        new_paulis = self._paulis[key]
        new_phases = self._phases[key]

        return PhasedPauliArray(new_paulis, new_phases)

    def __setitem__(self, key, value: "PhasedPauliArray"):
        if isinstance(value, PhasedPauliArray):
            self._phases[key] = value._phases
            self._paulis[key] = value._paulis
        else:
            raise ValueError()

    def __str__(self):
        return f"PhasedPauliArray: num_qubits = {self.num_qubits}, shape = {str(self.shape)}, ..."

    def __eq__(self, other: "PhasedPauliArray") -> "np.ndarray[np.bool]":
        """
        Checks element-wise if the other PhasedPauliArray is equal.

        Args:
            other (PhasedPauliArray): An other PhasedPauliArray. Must be broadcastable

        Returns:
            "np.ndarray[np.bool]": _description_
        """
        eq_paulis = self.paulis == other.paulis
        eq_phases = np.isclose(self.phases, other.phases)

        return np.logical_and(eq_paulis, eq_phases)

    def _mul(self, other: "PhasedPauliArray") -> "PhasedPauliArray":
        if isinstance(other, Number):
            return self.mul_phases(other)
        elif isinstance(other, PhasedPauliArray):
            return self.compose_phased_pauli_array(other)

        return NotImplemented

    __mul__ = __rmul__ = _mul

    def copy(self) -> "PhasedPauliArray":
        """
        Returns a copy of the PhasedPauliArray.

        Returns:
            PhasedPauliArray: Copied PhasedPauliArray.
        """
        return PhasedPauliArray(self._paulis.copy(), self._phases.copy())

    def adjoint(self) -> "PhasedPauliArray":

        new_phases = np.choose([0, 3, 2, 1], self.phases)

        return PhasedPauliArray(self.paulis, new_phases)

    def reshape(self, shape: Tuple[int, ...]) -> "PhasedPauliArray":
        """
        Reshape the PhasedPauliArray

        Args:
            shape (tuple[int]): New shape

        Returns:
            PhasedPauliArray: Reshaped PhasedPauliArray
        """

        # TODO check number of dimensions in shape

        new_phases = self._phases.reshape(shape)
        new_paulis = self._paulis.reshape(shape)

        return PhasedPauliArray(new_paulis, new_phases)

    def flatten(self) -> "PhasedPauliArray":
        """
        Returns a copy of the PhasedPauliArray flattened into one dimension.

        Returns:
            PhasedPauliArray: A flattened copy of the current PhasedPauliArray.
        """

        shape = (np.prod(self.shape, dtype=int),)

        return self.reshape(shape)

    def squeeze(self) -> "PhasedPauliArray":
        """
        Returns a PhasedPauliArray with axes of length one removed.

        Returns:
            PhasedPauliArray: The squeezed PhasedPauliArray.
        """
        new_paulis = self.paulis.squeeze()
        new_phases = self.phases.squeeze()

        return PhasedPauliArray(new_paulis, new_phases)

    def remove(self, index: int) -> "PhasedPauliArray":
        """
        Returns a PhasedPauliArray with removed item at given index.

        Args:
            index (int): Index of element to remove.

        Returns:
            PhasedPauliArray: PhasedPauliArray with removed item at given index.
        """
        new_paulis = self.paulis.remove(index)
        new_phases = self.phases.remove(index)

        return PhasedPauliArray(new_paulis, new_phases)

    def extract(self, condition: Union[NDArray, list]) -> "PhasedPauliArray":
        """
        Return the Pauli strings from the PhasedPauliArray object that satisfy some condition.

        Args:
          condition (Union[NDArray, list]): An array whose nonzero or True entries indicate the Pauli strings of PhasedPauliArray to extract.

        Returns:
            PhasedPauliArray: A new PhasedPauliArray object containing the extracted Pauli strings.

        Raises:
            ValueError: If the shape of the condition array is not equal to shape of the PhasedPauliArray.
        """
        if isinstance(condition, list):
            condition = np.array(condition, dtype=bool)

        if condition.shape != self.shape:
            raise ValueError("The condition array must have the same shape as the weighted Paulis.")

        new_phases = self.phases[condition]
        new_paulis = self.paulis[condition]

        if len(new_phases) == 0:
            return PhasedPauliArray.empty(self.num_qubits)

        return PhasedPauliArray(new_paulis, new_phases)

    def take_qubits(self, indices: Union["np.ndarray[np.int]", range, int]) -> "PhasedPauliArray":
        if isinstance(indices, int):
            indices = np.array([indices], dtype=int)

        new_phases = self.phases.copy()
        new_paulis = self.paulis.take_qubits(indices)

        return PhasedPauliArray(new_paulis, new_phases)

    def compress_qubits(self, condition: "np.ndarray[np.bool]") -> "PhasedPauliArray":
        new_phases = self.phases.copy()
        new_paulis = self.paulis.compress_qubits(condition)

        return PhasedPauliArray(new_paulis, new_phases)

    def compose(self, other: Any) -> Any:

        if isinstance(other, PhasedPauliArray):
            return self.compose_phased_pauli_array(other)

        return NotImplemented

    def compose_phased_pauli_array(self, other: "PhasedPauliArray") -> "PhasedPauliArray":
        new_paulis, phases = self._paulis.compose_pauli_array(other.paulis)
        new_phases = np.mod(self._phases + other.phases + phases, 4)

        return PhasedPauliArray(new_paulis, new_phases)

    def tensor(self, other: Any) -> Any:

        if isinstance(other, PhasedPauliArray):
            return self.tensor_phased_pauli_array(other)

        return NotImplemented

    def tensor_phased_pauli_array(self, other: "PhasedPauliArray") -> "PhasedPauliArray":
        new_paulis = self.paulis.tensor_pauli_array(other.paulis)
        new_phases = np.mod(self.phases + other.phases, 4)

        return PhasedPauliArray(new_paulis, new_phases)

    def commute_with(self, other: "PhasedPauliArray") -> "np.ndarray[np.bool]":
        return self.paulis.commute_with(other.paulis)

    def bitwise_commute_with(self, other: "PhasedPauliArray") -> "np.ndarray[np.bool]":
        return self.paulis.bitwise_commute_with(other.paulis)

    def inspect(self) -> str:
        if self.ndim == 0:
            return "Empty PauliArray"

        if self.ndim == 1:
            label_table = self.label_table_1d(self.to_labels(), self.phases)
            return f"PauliArray\n{label_table}"

        if self.ndim == 2:
            label_table = self.label_table_2d(self.to_labels(), self.phases)
            return f"PauliArray\n{label_table}"

        label_table = self.label_table_nd(self.to_labels(), self.phases)
        return f"PauliArray\n{label_table}"

    def clifford_conjugate(self, clifford: "Operator", inplace: bool = True) -> "PhasedPauliArray":
        """
        Performs a Clifford transformation.

        Args:
            clifford (Operator) : Must represent a Clifford transformation with the correct number of qubits.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PhasedPauliArray: The transformed PhasedPauliArray
        """

        return NotImplemented

        new_paulis, factors = clifford.clifford_conjugate_pauli_array_old(self.paulis)
        new_phases = self.phases * factors
        if inplace:
            self._paulis = new_paulis
            self._phases = new_phases

            return self

        return PhasedPauliArray(new_paulis, new_phases)

    def expectation_values_from_paulis(
        self, paulis_expectation_values: NDArray[np.float64]
    ) -> "np.ndarray[np.complex]":
        """
        Returns the PhasedPauliArray expectation value given the expectation values of the Paulis.

        Args:
            paulis_expectation_values (NDArray[float]): _description_

        Returns:
            NDArray: _description_
        """

        assert np.all(paulis_expectation_values.shape == self.shape)

        phase_factors = np.choose(self.phases, [1, -1j, -1, 1j])

        return phase_factors * paulis_expectation_values

    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float64]) -> "np.ndarray[np.complex]":
        """
        Returns the PhasedPauliArray covariances given the covariances of the Paulis.

        Args:
            paulis_covariances (NDArray[float]): _description_

        Returns:
            NDArray: _description_
        """

        return NotImplemented

        assert np.all(paulis_covariances.shape == (self.shape + self.shape))

        flat_phases = self.phases.flatten()
        flat_paulis_covariances = paulis_covariances.reshape((self.size, self.size))

        flat_wpaulis_covariances = flat_phases[:, None] * flat_phases[None, :].conj() * flat_paulis_covariances

        return flat_wpaulis_covariances.reshape((self.shape + self.shape))

    def update_phases(self, new_phases):
        assert np.all(self.phases.shape == new_phases.shape)

        self._phases = new_phases.copy()

    def is_diagonal(self) -> "np.ndarray[np.bool]":
        """
        Checks if the Pauli strings are diagonal i.e. if all Pauli strings are I or Z.

        Returns:
            NDArray[bool]: True if the Pauli string is diagonal, False otherwise.
        """
        return self._paulis.is_diagonal()

    def to_labels(self) -> "np.ndarray[np.str]":
        """
        Returns the labels of all zx strings.

        Returns:
            "np.ndarray[np.str]": An array containing the labels of all Pauli strings.
        """

        pauli_labels = self.paulis.to_labels()
        phase_labels = np.array(["  ", "-i", " -", " i"])[self.phases]

        labels = np.char.add(phase_labels, pauli_labels)

        return labels

    def to_matrices(self) -> NDArray:
        """
        Returns the PhasedPauliArray as a numpy matrix.

        Returns:
            matrices (NDArray): An ndarray of shape self.shape + (n**2, n**2).
        """

        phase_factors = np.choose(self.phases, [1, -1j, -1, 1j])

        return phase_factors[..., None, None] * self.paulis.to_matrices()

    @classmethod
    def new(cls, shape: Tuple[int, ...], num_qubits: int) -> "PhasedPauliArray":
        phases = np.zeros(shape, dtype=np.uint)
        paulis = pa.PauliArray.identities(shape, num_qubits)

        return PhasedPauliArray(paulis, phases)

    @classmethod
    def empty(cls, num_qubits: int) -> "PhasedPauliArray":
        """
        Returns an empty PhasedPauliArray with the number of qubits already set.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            PhasedPauliArray: An empty PhasedPauliArray
        """
        phases = np.zeros((), dtype=np.uint)
        paulis = pa.PauliArray.identities((), num_qubits)

        return PhasedPauliArray(paulis, phases)

    @classmethod
    def random(cls, shape: Tuple[int, ...], num_qubits: int) -> "PhasedPauliArray":
        """
        Creates a PauliArray of a given shape and number of qubits filled with random Pauli strings.

        Args:
            shape (_type_): Shape of new PauliArray.
            num_qubits (_type_): Number of qubits of new PauliArray.

        Returns:
            new_PauliArray (PauliArray): The PauliArray created.
        """
        random_paulis = pa.PauliArray.random(shape, num_qubits)
        random_phases = np.random.choice(range(4), shape)

        return PhasedPauliArray(random_paulis, random_phases)

    @classmethod
    def from_labels(cls, labels) -> "PhasedPauliArray":

        phases, pauli_labels = cls.split_phases_pauli_labels(labels)

        paulis = pa.PauliArray.from_labels(pauli_labels)

        return PhasedPauliArray(paulis, phases)

    @staticmethod
    def split_phase_pauli_label(label):

        ms = re.match("([+-]{0,1})([1]{0,1})([ij]{0,1})([IXYZ]*)", label)

        if not bool(ms):
            raise ValueError("A label cannot be interpreted")

        sign_bit = ms.group(1) == "-"
        imag_bit = len(ms.group(3)) > 0
        pauli_label = ms.group(4)

        phase = [0, 3, 2, 1][imag_bit + 2 * sign_bit]

        return phase, pauli_label

    @staticmethod
    def split_phases_pauli_labels(labels):

        labels = np.atleast_1d(np.array(labels, dtype=str))

        num_qubits = len(PhasedPauliArray.split_phase_pauli_label(labels.flat[0])[1])

        phases = np.zeros(labels.shape, dtype=np.uint)
        pauli_labels = np.zeros(labels.shape, dtype=f"U{num_qubits}")

        for idx, label in np.ndenumerate(labels):
            phases[idx], pauli_labels[idx] = PhasedPauliArray.split_phase_pauli_label(label)

        return phases, pauli_labels

    @classmethod
    def from_z_strings_and_x_strings_and_phases(
        cls,
        z_strings: "np.ndarray[np.bool]",
        x_strings: "np.ndarray[np.bool]",
        phases: "np.ndarray[np.uint]",
    ) -> "PhasedPauliArray":

        paulis = pa.PauliArray(z_strings, x_strings)

        return PhasedPauliArray(paulis, phases)

    @classmethod
    def from_paulis(cls, paulis: pa.PauliArray) -> "PhasedPauliArray":
        phases = np.zeros(paulis.shape, dtype=np.uint)

        return PhasedPauliArray(paulis.copy(), phases)

    def to_npz(self, filename):
        with open(filename, "wb") as f:
            np.save(f, self.paulis.zx_strings)
            np.save(f, self.phases)

    @classmethod
    def from_npz(cls, filename) -> "PhasedPauliArray":
        with open(filename, "rb") as f:
            zx_strings = np.load(f)
            phases = np.load(f)

        return PhasedPauliArray(pa.PauliArray.from_zx_strings(zx_strings), phases)

    @staticmethod
    def label_table_1d(pauli_labels, phase_labels) -> str:

        return NotImplemented

        pauli_str_len = len(max(pauli_labels, key=len))

        row_strs = []
        for label, weight in zip(pauli_labels, phase_labels):
            row_strs.append(f"({weight.real:+7.4f} {weight.imag:+7.4f}j) {label:{pauli_str_len}s}")

        return "\n".join(row_strs)

    @staticmethod
    def label_table_2d(labels, phases) -> str:

        return NotImplemented

        pauli_str_len = len(max(labels, key=len))

        row_strs = []
        for i in range(labels.shape[0]):
            col_strs = []
            for label, weight in zip(labels[i, :], phases[i, :]):
                col_strs.append(f"({weight.real:+7.4f} {weight.imag:+7.4f}j) {label:{pauli_str_len}s}")
            row_strs.append("  ".join(col_strs))

        return "\n".join(row_strs)

    @staticmethod
    def label_table_nd(labels, phases) -> str:

        return NotImplemented

        slice_strs = []
        for idx in np.ndindex(labels.shape[:-2]):
            slice_str = "Slice (" + ",".join([str(i) for i in idx]) + ",:,:)\n"
            slice_str += PhasedPauliArray.label_table_2d(labels[idx], phases[idx])
            slice_strs.append(slice_str)

        return "\n".join(slice_strs)


def broadcast_to(ppaulis: PhasedPauliArray, shape: Tuple[int, ...]) -> "PhasedPauliArray":
    """
    Returns the given PhasedPauliArray broadcasted to a given shape.

    Args:
        paulis (PhasedPauliArray): PhasedPauliArray to broadcast.
        shape (Tuple[int, ...]): Shape to broadcast to.

    Returns:
        new_pauli_array (PhasedPauliArray): The PhasedPauliArray with a new shape.
    """

    new_paulis = pa.broadcast_to(ppaulis.paulis, shape)
    new_phases = np.broadcast_to(ppaulis.phases, shape)

    return PhasedPauliArray(new_paulis, new_phases)


def expand_dims(ppaulis: PhasedPauliArray, axis=Union[int, Tuple[int, ...]]) -> "PhasedPauliArray":
    """
    Expands the shape of a PhasedPauliArray.

    Inserts a new axis that will appear at the axis position in the expanded array shape.

    Args:
        paulis (PhasedPauliArray): The PhasedPauliArray to expand.
        axis (Union[int, Tuple[int, ...]]): The axis upon which expand the PhasedPauliArray.

    Returns:
        expanded_pauli_array (PhasedPauliArray) : The expanded PhasedPauliArray.
    """

    new_paulis = pa.expand_dims(ppaulis.paulis, axis)
    new_phases = np.expand_dims(ppaulis.phases, axis)

    return PhasedPauliArray(new_paulis, new_phases)


def commutator(wpaulis_1: PhasedPauliArray, wpaulis_2: PhasedPauliArray) -> PhasedPauliArray:
    """
    Returns the commutator of the two PhasedPauliArray parameters.

    Args:
        wpaulis_1 (PhasedPauliArray): PhasedPauliArray to calculate commmutator with.
        wpaulis_2 (PhasedPauliArray): Other PhasedPauliArray to calculate commmutator with.

    Returns:
        commutator_pauli_array (PhasedPauliArray): PhasedPauliArray containing the commutators.
    """

    assert is_broadcastable(wpaulis_1.shape, wpaulis_2.shape)

    commutators = wpaulis_1.compose_phased_pauli_array(wpaulis_2)
    do_commute = wpaulis_1.commute_with(wpaulis_2)

    commutators.paulis.x_strings[do_commute] = 0
    commutators.paulis.z_strings[do_commute] = 0

    commutators._phases *= ~do_commute * 2

    return commutators


def anticommutator(wpaulis_1: PhasedPauliArray, wpaulis_2: PhasedPauliArray) -> PhasedPauliArray:
    assert is_broadcastable(wpaulis_1.shape, wpaulis_2.shape)

    anticommutators = wpaulis_1.compose_phased_pauli_array(wpaulis_2)
    do_commute = wpaulis_1.commute_with(wpaulis_2)

    anticommutators.paulis.x_strings[~do_commute] = 0
    anticommutators.paulis.z_strings[~do_commute] = 0

    anticommutators._phases *= do_commute * 2

    return anticommutators


def concatenate(wpauli_arrays: Tuple[PhasedPauliArray, ...], axis: int) -> PhasedPauliArray:
    """
    Concatenated multiple WeightedPauliArrays.

    Args:
        paulis (List[PauliArray]): WeightedPauliArrays to concatenate.
        axis (int): The axis along which the arrays will be joined.

    Returns:
        PhasedPauliArray: The concatenated WeightedPauliArrays.
    """

    assert is_concatenatable(wpauli_arrays, axis)

    weights_list = tuple(ppaulis.phases for ppaulis in wpauli_arrays)
    paulis_list = tuple(ppaulis.paulis for ppaulis in wpauli_arrays)

    new_phases = np.concatenate(weights_list, axis)
    new_paulis = pa.concatenate(paulis_list, axis)

    return PhasedPauliArray(new_paulis, new_phases)


def swapaxes(ppaulis: PhasedPauliArray, axis1: int, axis2: int):
    """
    Swap axes of a PhasedPauliArray

    Args:
        paulis (PhasedPauliArray): The PhasedPauliArray
        axis1 (int): Original axis position
        axis2 (int): Target axis position

    Returns:
        PhasedPauliArray: The WeightedPauliArrays with axes swaped.
    """

    assert axis1 < ppaulis.ndim
    assert axis2 < ppaulis.ndim

    new_paulis = pa.swapaxes(ppaulis.paulis, axis1, axis2)
    new_phases = np.swapaxes(ppaulis.phases, axis1, axis2)

    return PhasedPauliArray(new_paulis, new_phases)


def moveaxis(ppaulis: PhasedPauliArray, source: int, destination: int):
    """
    Move an axis of a PhasedPauliArray

    Args:
        paulis (PhasedPauliArray): The PhasedPauliArray
        axis1 (int): Original axis position
        axis2 (int): Target axis position

    Returns:
        PhasedPauliArray: The WeightedPauliArrays with axis moved.
    """

    assert source < ppaulis.ndim
    assert destination < ppaulis.ndim

    new_paulis = pa.moveaxis(ppaulis.paulis, source, destination)
    new_phases = np.moveaxis(ppaulis.phases, source, destination)

    return PhasedPauliArray(new_paulis, new_phases)
