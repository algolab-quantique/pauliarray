from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Self, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pauliarray.binary import bit_operations as bitops
from pauliarray.binary import symplectic
from pauliarray.utils import labels_to_table

#
from pauliarray.utils.array_operations import broadcast_shape, is_broadcastable, is_concatenatable

if TYPE_CHECKING:
    from pauliarray.pauli.operator import Operator
    from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1
    from pauliarray.pauli.weighted_pauli_array import WeightedPauliArray


PAULI_LABELS = "IZXY"
PAULI_TO_ZX_BITS = {
    "I": (False, False),
    "X": (False, True),
    "Y": (True, True),
    "Z": (True, False),
}
ZX_BITS_TO_PAULI_LABEL = {
    (False, False): "I",
    (False, True): "X",
    (True, True): "Y",
    (True, False): "Z",
}
ZX_BITS_TO_PAULI_MAT = {
    (False, False): np.array([[1, 0], [0, 1]]),
    (False, True): np.array([[0, 1], [1, 0]]),
    (True, True): np.array([[0, -1j], [1j, 0]]),
    (True, False): np.array([[1, 0], [0, -1]]),
}


class PauliArray(object):
    """
    Defines an array of Pauli strings.
    """

    def __init__(self, z_strings: "np.ndarray[np.bool]", x_strings: "np.ndarray[np.bool]"):

        z_strings = np.atleast_2d(z_strings)
        x_strings = np.atleast_2d(x_strings)

        assert np.all(z_strings.shape == x_strings.shape)

        assert z_strings.dtype == np.dtype(np.bool_)
        assert x_strings.dtype == np.dtype(np.bool_)

        self._z_strings = z_strings
        self._x_strings = x_strings

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self._z_strings.shape[-1]

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the object.

        Returns:
            Tuple[int, ...]: The shape of the object.
        """
        return self._z_strings.shape[:-1]

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
        Returns the total number of elements in the PauliArray.

        Returns:
            int: The total number of elements in the PauliArray.
        """
        return np.prod(self.shape)

    @property
    def paulis(self) -> Self:
        return self

    @property
    def x_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the X bits.

        Returns:
            "np.ndarray[np.bool]": The X bits.
        """
        return self._x_strings

    @property
    def z_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the Z bits.

        Returns:
            "np.ndarray[np.bool]": The Z bits.
        """
        return self._z_strings

    @property
    def zx_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the combined Z and X bits.

        Returns:
            "np.ndarray[np.bool]": The combined Z and X bits.
        """
        return symplectic.merge_zx_strings(self._z_strings, self._x_strings)

    @property
    def xz_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the combined X and Z bits.

        Returns:
            "np.ndarray[np.bool]": The combined X and Z bits.
        """
        return symplectic.merge_zx_strings(self._x_strings, self._z_strings)

    @property
    def num_ids(self) -> "np.ndarray[np.int]":
        """
        Returns the number of identity operators.

        Returns:
            "np.ndarray[np.int]": The number of identity operators.
        """
        return np.sum(~np.logical_or(self._z_strings, self._x_strings), axis=-1)

    @property
    def num_non_ids(self) -> "np.ndarray[np.int]":
        """
        Returns the number of non-identity operators.

        Returns:
            "np.ndarray[np.int]": The number of non-identity operators.
        """
        return np.sum(np.logical_or(self._z_strings, self._x_strings), axis=-1)

    def replace_paulis(self, new_paulis: "PauliArray", inplace=False) -> Self:
        """
        Replace the Paulis in the object by the new ones.

        Args:
            new_paulis (PauliArray): The new Paulis
            inplace (bool, optional): If True replace inside the current instance. If False returns a new instance. Defaults to False.

        Returns:
            PauliArray: The object with replaced Paulis
        """

        assert new_paulis.shape == self.shape

        if inplace:
            self._z_strings = new_paulis.z_strings.copy()
            self._x_strings = new_paulis.x_strings.copy()
            return self

        return PauliArray(new_paulis.z_strings.copy(), new_paulis.x_strings.copy())

    def clifford_transform_paulis(
        self, clifford_fct: Callable, *args, inplace=False
    ) -> Tuple[Self, "np.ndarray[np.complex]"]:
        """
        Transform the Paulis using a Clifford transformation (from transformation.cliffords)

        Args:
            clifford_fct (Callable): A Clifford function (from transformation.cliffords)
            *args: The arguments of the Clifford function, such as the qubits on which to apply.
            inplace (bool, optional): If True replace the existing Paulis. Defaults to False.

        Returns:
            PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        new_paulis, phases = clifford_fct(self, *args)
        if inplace:
            self.replace_paulis(new_paulis)
            return self, phases

        return new_paulis, phases

    def __getitem__(self, key):
        # TODO check number of dimensions in key
        new_z_strings = self._z_strings[key]
        new_x_strings = self._x_strings[key]

        return PauliArray(new_z_strings, new_x_strings)

    def __setitem__(self, key, value: "PauliArray"):
        if isinstance(value, PauliArray):
            self._z_strings[key] = value.z_strings
            self._x_strings[key] = value.x_strings
        else:
            raise TypeError("Value should be of type PauliArray.")

    def __str__(self):
        return f"PauliArray: num_qubits = {self.num_qubits}, shape = {self.shape}, ..."

    def __eq__(self, other: Self) -> "np.ndarray[np.bool]":
        """
        Checks element-wise if the other PauliArray is equal.

        Args:
            other (PauliArray): An other PauliArray. Must be broadcastable

        Returns:
            "np.ndarray[np.bool]": _description_
        """

        return np.all(np.logical_and((self.z_strings == other.z_strings), (self.x_strings == other.x_strings)), axis=-1)

    def copy(self) -> Self:
        """
        Returns a copy of the PauliArray.

        Returns:
            PauliArray: Copied PauliArray.
        """
        return PauliArray(self.z_strings.copy(), self.x_strings.copy())

    def reshape(self, shape: Tuple[int, ...]) -> Self:
        """
        Returns a PauliArray with a new shape.

        Args:
            shape (tuple[int]): Tuple containing the new shape e.g. for 2x2 matrix: shape = (2,2)

        Returns:
            PauliArray: The PauliArray object with the new shape.
        """

        # TODO check number of dimensions in shape

        new_shape = shape + (self.num_qubits,)
        new_z_strings = self._z_strings.reshape(new_shape)
        new_x_strings = self._x_strings.reshape(new_shape)

        return PauliArray(new_z_strings, new_x_strings)

    def flatten(self) -> Self:
        """
        Returns a copy of the PauliArray flattened into one dimension.

        Returns:
            PauliArray: A flattened copy of the current PauliArray.
        """
        shape = (np.prod(self.shape),)

        return self.reshape(shape)

    def squeeze(self) -> Self:
        """
        Returns a PauliArray with axes of length one removed.

        Returns:
            PauliArray: The squeezed PauliArray.
        """
        new_z_strings = self._z_strings.squeeze()
        new_x_strings = self._x_strings.squeeze()

        return PauliArray(new_z_strings, new_x_strings)

    def remove(self, index: int) -> Self:
        """
        Returns a PauliArray with removed item at given index.

        Args:
            index (int): Index of element to remove.

        Returns:
            PauliArray: PauliArray with removed item at given index.
        """
        new_z_strings = self._z_strings.remove(index)
        new_x_strings = self._x_strings.remove(index)

        return PauliArray(new_z_strings, new_x_strings)

    def extract(self, condition: Union[NDArray, list]) -> Self:
        """
        Return the Pauli strings from the PauliArray object that satisfy some condition.

        Args:
          condition (Union[NDArray, list]): An array whose nonzero or True entries indicate the Pauli strings of PauliArray to extract.

        Returns:
            PauliArray: A new PauliArray object containing the extracted Pauli strings.

        Raises:
            ValueError: If the shape of the condition array is not equal to shape of the PauliArray.
        """
        if isinstance(condition, list):
            condition = np.array(condition, dtype=bool)

        if condition.shape != self.shape:
            raise ValueError("The condition array must have the same shape as the PauliArray.")

        new_x_strings = self.x_strings[condition]
        new_z_strings = self.z_strings[condition]

        if new_x_strings.size == 0:
            return PauliArray.identities((), self.num_qubits)

        return PauliArray(new_z_strings, new_x_strings)

    def take_qubits(self, indices: Union["np.ndarray[np.int]", range, int], inplace: bool = True) -> Self:
        """
        Return the Pauli strings for a subset of qubits, ignoring the other. Using indices.

        Args:
            indices (Union["np.ndarray[np.int]", range, int]): Index or array of indices of the qubits to return.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: PauliArray with a reduced number of qubits.
        """

        if not inplace:
            return self.copy().take_qubits(indices)

        if isinstance(indices, int):
            indices = np.array([indices], dtype=int)

        new_z_strings = self.z_strings.take(indices, axis=-1)
        new_x_strings = self.x_strings.take(indices, axis=-1)

        return PauliArray(new_z_strings, new_x_strings)

    def compress_qubits(self, condition: "np.ndarray[np.bool]", inplace: bool = True) -> Self:
        """
        Return the Pauli strings for a subset of qubits, ignoring the other. Using a mask.

        Args:
            condition ("np.ndarray[np.bool]"): Array that selects which qubit to keep. Must be on length equal to the number of qubits.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: PauliArray with a reduced number of qubits.
        """

        if not inplace:
            return self.copy().compress_qubits(condition)

        new_z_strings = self.z_strings.compress(condition, axis=-1)
        new_x_strings = self.x_strings.compress(condition, axis=-1)

        return PauliArray(new_z_strings, new_x_strings)

    def reorder_qubits(self, qubit_order: List[int], inplace: bool = True) -> Self:
        """
        Reorder the qubits.

        Args:
            qubit_order: The new qubits order. Must contain each qubit index once.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: The transformed PauliArray
        """

        assert np.all(np.sort(qubit_order) == np.arange(self.num_qubits))

        if not inplace:
            return self.copy().reorder_qubits(qubit_order)

        self.z_strings[..., :] = self.z_strings[..., qubit_order]
        self.x_strings[..., :] = self.x_strings[..., qubit_order]

        return self

    def compose(self, other: Any) -> Any:

        if isinstance(other, PauliArray):
            return self.compose_pauli_array(other)

        return NotImplemented

    def compose_pauli_array(self, other: Self) -> Tuple[Self, "np.ndarray[np.complex]"]:
        """
        Performs an element-wise composition with an other PauliArray.

        Args:
            other (PauliArray): The PauliArray to compose with. Must be broadcastable.

        Returns:
            PauliArray: The result of the composition.
            "np.ndarray[np.unit]" : Mod 4 phases resulting from the composition.
        """
        assert self.num_qubits == other.num_qubits
        assert is_broadcastable(self.shape, other.shape)

        new_z_strings, new_x_strings, phases = symplectic.zx_composition_with_phase(self.zx_strings, other.zx_strings)

        return PauliArray(new_z_strings, new_x_strings), phases

    def tensor(self, other: Any) -> Any:

        if isinstance(other, PauliArray):
            return self.tensor_pauli_array(other)

        return NotImplemented

    def tensor_pauli_array(self, other: Self) -> Self:
        """
        Performs a tensor product, element-wise with an other PauliArray.

        Args:
            other (PauliArray): The PauliArray to multiply with. Must be broadcastable.

        Returns:
            PauliArray: The result of the tensor product.
        """
        new_z_strings = np.concatenate((self.z_strings, other.z_strings), axis=-1)
        new_x_strings = np.concatenate((self.x_strings, other.x_strings), axis=-1)

        return PauliArray(new_z_strings, new_x_strings)

    def flip_zx(self) -> Self:
        """
        Returns a copy of the PauliArray with x strings as z strings, and vice versa.

        Returns:
            PauliArray: The PauliArray with flipped z and x strings.
        """
        return PauliArray(self.x_strings.copy(), self.z_strings.copy())

    def commute_with(self, other: Self) -> "np.ndarray[np.bool]":
        """
        Returns True if the elements of PauliArray commutes with the elements of PauliArray passed as parameter,
        returns False otherwise.

        Args:
            other (PauliArray): The PauliArray to check commutation with.

        Returns:
            "np.ndarray[np.bool]": An array of bool set to true for commuting Pauli string, and false otherwise.
        """

        return ~np.mod(symplectic.dot(self.zx_strings, other.zx_strings), 2).astype(np.bool_)

    def bitwise_commute_with(self, other: Self) -> "np.ndarray[np.bool]":
        """
        Returns True if the elements of PauliArray commutes bitwise with the elements of
        PauliArray passed as parameter, returns False otherwise.

        Args:
            other (PauliArray): The other PauliArray to verify bitwise commutation with.

        Returns:
            "np.ndarray[np.bool]": An array of bool set to true for bitwise commuting Pauli string, and false otherwise.
        """

        ovlp_1 = self.z_strings * other.x_strings
        ovlp_2 = self.x_strings * other.z_strings

        return np.all(~np.logical_xor(ovlp_1, ovlp_2), axis=-1)

    def traces(self) -> NDArray:
        """
        Return the traces of the Pauli Strings which are 2^n if Identity and 0 otherwise.

        Returns:
            "np.ndarray[np.int]": Traces of the Pauli Strings
        """

        return 2**self.num_qubits * (self.num_ids == self.num_qubits)

    def generators(self) -> Self:
        """
        Finds a set of linearly independant PauliString which can be multiplied together to generate every PauliStirng
        in self.

        Returns:
            PauliArray: The generators
        """

        generator_zx_strings = bitops.row_space(self.flatten().zx_strings)

        generators = PauliArray(
            generator_zx_strings[..., : self.num_qubits], generator_zx_strings[..., self.num_qubits :]
        )

        return generators

    def generators_with_map(self) -> Tuple["PauliArray", "np.ndarray[np.bool]"]:
        """
        Finds a set of linearly independant PauliString which can be multiplied together to generate every PauliStirng
        in self. Alse returns a matrix identifying which generators are involved in each PauliString in self.

        Returns:
            PauliArray: The generators
            NDArray: Element [idx,j] = True if generator j is used to construct self[idx]
        """

        generators = self.generators()

        combinaison_map = np.any(self.zx_strings[..., None, :] * generators.zx_strings[None, :, :], axis=-1)

        return generators, combinaison_map

    def inspect(self) -> str:
        """
        Returns an inspection string showing all labels of the PauliArray.

        Returns:
            str: The inspection string.
        """
        if self.ndim == 0:
            return "Empty PauliArray"

        if self.ndim == 1:
            label_table = labels_to_table.label_table_1d(self.to_labels())
            return f"PauliArray\n{label_table}"

        if self.ndim == 2:
            label_table = labels_to_table.label_table_2d(self.to_labels())
            return f"PauliArray\n{label_table}"

        label_table = labels_to_table.label_table_nd(self.to_labels())

        return f"PauliArray\n{label_table}"

    def clifford_conjugate(
        self, clifford: "Operator", inplace: bool = True
    ) -> Tuple["PauliArray", "np.ndarray[np.complex]"]:
        """
        Performs a Clifford transformation.

        Args:
            clifford (Operator) : Must represent a Clifford transformation with the correct number of qubits.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        new_paulis, factors = clifford.clifford_conjugate_pauli_array_old(self)
        if inplace:
            self._z_strings = new_paulis.z_strings
            self._x_strings = new_paulis.x_strings
            return self, factors

        return new_paulis, factors

    def expectation_values_from_paulis(self, paulis_expectation_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the PauliArray expectation value given the expectation values of the Paulis. More useful for other classes, but still here for uniformity.

        Args:
            paulis_expectation_values (NDArray[float]): The expectation values of the underlying PauliArray. Must be of the same shape as self.

        Returns:
            NDArray: The expectation values.
        """
        assert np.all(paulis_expectation_values.shape == self.shape)

        return paulis_expectation_values

    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the PauliArray covariances given the covariances of the Paulis. More useful for other classes, but still here for uniformity.

        Args:
            paulis_covariances (NDArray[float]): The covariance array of the underlying PauliArray. Must be of shape self.shape + self.shape

        Returns:
            NDArray: The covariance array.
        """
        assert np.all(paulis_covariances.shape == (self.shape + self.shape))

        return paulis_covariances

    def is_diagonal(self) -> "np.ndarray[np.bool]":
        """
        Checks if the Pauli strings are diagonal i.e. if all Pauli strings are I or Z.

        Returns:
            NDArray[bool]: True if the Pauli string is diagonal, False otherwise.
        """
        return ~np.any(self.x_strings, axis=-1)

    def is_identity(self) -> "np.ndarray[np.bool]":
        """
        Checks if the Pauli strings are identities i.e. if all Pauli strings are I.

        Returns:
            NDArray[bool]: True if the Pauli string is identity, False otherwise.
        """
        return ~np.logical_or(np.any(self.x_strings, axis=-1), np.any(self.z_strings, axis=-1))

    def to_labels(self) -> "np.ndarray[np.str]":
        """
        Returns the labels of all zx strings.

        Returns:
            "np.ndarray[np.str]": An array containing the labels of all Pauli strings.
        """
        labels = np.zeros(self.shape, dtype=f"U{self.num_qubits}")

        for idx in np.ndindex(*self.shape):
            labels[idx] = self.z_string_x_string_to_label(self.z_strings[idx], self.x_strings[idx])

        return labels

    def to_matrices(self) -> NDArray:
        """
        Returns the Pauli strings as a matrices.

        Returns:
            matrices (NDArray): An ndarray of shape self.shape + (n**2, n**2).
        """
        mat_shape = (2**self.num_qubits, 2**self.num_qubits)
        matrices = np.zeros(self.shape + mat_shape, dtype=complex)

        z_ints = bitops.strings_to_ints(self.z_strings)
        x_ints = bitops.strings_to_ints(self.x_strings)

        phase_powers = np.mod(bitops.dot(self.z_strings, self.x_strings), 4)
        phases = np.choose(phase_powers, [1, -1j, -1, 1j])

        for idx in np.ndindex(self.shape):
            one_matrix = self.matrix_from_zx_ints(z_ints[idx], x_ints[idx], self.num_qubits)
            matrices[idx] = one_matrix

        return phases[..., None, None] * matrices

    @staticmethod
    def sparse_matrix_from_zx_ints(z_int: int, x_int: int, num_qubits: int) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Builds the matrix representing the Pauli String encoded in a sparse notation.

        Args:
            z_int (int): Integer which binary representation defines the z part of a Pauli String.
            x_int (int): Integer which binary representation defines the x part of a Pauli String.
            num_qubits (int): Length of the Pauli String.

        Returns:
            row_ind (NDArray): The row indices of returned matrix elements.
            col_ind (NDArray): The column indices of returned matrix elements.
            matrix_elements (NDArray): The matrix elements.
        """
        dim = 2**num_qubits

        row_ind = np.arange(dim, dtype=np.int64)

        col_ind = np.bitwise_xor(row_ind, x_int)
        matrix_elements = np.array([1 - 2 * (bin(i).count("1") % 2) for i in np.bitwise_and(row_ind, z_int)])

        return row_ind, col_ind, matrix_elements

    @staticmethod
    def matrix_from_zx_ints(z_int: int, x_int: int, num_qubits: int) -> NDArray:
        """
        Builds the matrix representing the Pauli String.

        Args:
            z_int (int): Integer which binary representation defines the z part of a Pauli String.
            x_int (int): Integer which binary representation defines the x part of a Pauli String.
            num_qubits (int): Length of the Pauli String.

        Returns:
            NDArray: The matrix reprensetating the Pauli String.
        """
        row_ind, col_ind, matrix_elements = PauliArray.sparse_matrix_from_zx_ints(z_int, x_int, num_qubits)

        dim = 2**num_qubits
        mat_shape = (dim, dim)

        matrix = np.zeros(mat_shape, dtype=complex)
        matrix[row_ind, col_ind] = matrix_elements

        return matrix

    @classmethod
    def from_labels(cls, labels: Union[list[str], "np.ndarray[np.str]"]) -> Self:
        """
        Creates a PauliArray from a labels using IXYZ.

        Args:
            labels (Union[list[str], "np.ndarray[np.str]"]): The list of labels.

        Returns:
            new_pauli_array (PauliArray): The PauliArray created from labels.
        """
        if type(labels) not in (list, np.ndarray):
            labels = [labels]
        z_strings, x_strings = cls.labels_to_z_strings_x_strings(labels)

        return PauliArray(z_strings, x_strings)

    @classmethod
    def from_zx_strings(cls, zx_strings: "np.ndarray[np.bool]") -> Self:
        """
        Create a PauliArray from zx strings.

        Args:
            zx_strings ("np.ndarray[np.bool]"): Array where the last dimension size is an even integers (twice the number of qubits.)

        Returns:
            PauliArray: The created PauliArray .
        """
        assert (zx_strings.shape[-1] % 2) == 0
        num_qubits = zx_strings.shape[-1] // 2
        z_strings = zx_strings[..., :num_qubits]
        x_strings = zx_strings[..., num_qubits:]

        return PauliArray(z_strings, x_strings)

    @classmethod
    def identities(cls, shape: Tuple[int, ...], num_qubits: int) -> Self:
        """
        Creates a new PauliArray of a given shape and number of qubits filled with identities.

        Args:
            shape (_type_): The shape of new the PauliArray.
            num_qubits (_type_): The number of qubits of the new PauliArray.

        Returns:
            PauliArray: The created PauliArray .
        """
        new_shape = shape + (num_qubits,)

        z_strings = np.zeros(new_shape, dtype=np.bool_)
        x_strings = np.zeros(new_shape, dtype=np.bool_)

        return PauliArray(z_strings, x_strings)

    @classmethod
    def new(cls, shape: Tuple[int, ...], num_qubits: int) -> Self:

        return cls.identities(shape, num_qubits)

    @classmethod
    def random(cls, shape: Tuple[int, ...], num_qubits: int) -> Self:
        """
        Creates a PauliArray of a given shape and number of qubits filled with random Pauli strings.

        Args:
            shape (_type_): Shape of new PauliArray.
            num_qubits (_type_): Number of qubits of new PauliArray.

        Returns:
            new_PauliArray (PauliArray): The PauliArray created.
        """
        new_shape = shape + (num_qubits,)

        z_strings = np.random.choice([False, True], new_shape)
        x_strings = np.random.choice([False, True], new_shape)

        return PauliArray(z_strings, x_strings)

    @staticmethod
    def labels_to_z_strings_x_strings(
        labels: Union[list[str], "np.ndarray[np.str]"],
    ) -> Tuple["np.ndarray[np.bool]", "np.ndarray[np.bool]"]:
        """
        Returns z strings and x strings created from labels.

        Args:
            labels (Union[list[str], "np.ndarray[np.str]"]): The list of labels.

        Returns:
            "np.ndarray[np.bool]" : The z strings
            "np.ndarray[np.bool]" : The x strings
        """
        labels = np.atleast_1d(np.array(labels, dtype=str))

        num_qubits = len(labels[(0,) * labels.ndim])
        shape = labels.shape

        z_strings = np.zeros(shape + (num_qubits,), dtype=np.bool_)
        x_strings = np.zeros(shape + (num_qubits,), dtype=np.bool_)

        for idx in np.ndindex(*shape):
            z_strings[idx], x_strings[idx] = PauliArray.label_to_z_string_x_string(labels[idx])

        return z_strings, x_strings

    @staticmethod
    def label_to_z_string_x_string(label: str) -> Tuple["np.ndarray[np.bool]", "np.ndarray[np.bool]"]:
        """
        Returns the z and x strings corresponding to the label passed as parameter.

        Args:
            label (str): The label to convert to z_string and x_string.

        Returns:
            "np.ndarray[np.bool]" : The z strings
            "np.ndarray[np.bool]" : The x strings
        """
        label = label.upper()
        n_bits = len(label)

        z_string = np.zeros((n_bits,), dtype=bool)
        x_string = np.zeros((n_bits,), dtype=bool)

        for i, pauli_char in enumerate(reversed(label)):
            z_string[i], x_string[i] = PAULI_TO_ZX_BITS[pauli_char]

        return z_string, x_string

    @staticmethod
    def z_string_x_string_to_label(z_string: "np.ndarray[np.bool]", x_string: "np.ndarray[np.bool]") -> str:
        """
        Converts a pair of z string and x string into a label (IXYZ).

        Args:
            z_string ("np.ndarray[np.bool]"): Single z string
            x_string ("np.ndarray[np.bool]"): Single x string
        Returns:
            str: Label from the zx strings
        """
        pauli_choices = z_string + 2 * x_string

        label = ""
        for pauli in reversed(pauli_choices):
            label += PAULI_LABELS[pauli]

        return label


def argsort(paulis: PauliArray, axis: int = -1) -> "np.ndarray[np.int]":
    """
    Return indices which sorts the Pauli Strings.

    Returns:
        NDArray: Indices which sorts the Pauli Strings.
    """
    zx_ints = bitops.strings_to_ints(paulis.zx_strings)

    return np.argsort(zx_ints, axis)


def broadcast_to(paulis: PauliArray, shape: Tuple[int, ...]) -> Self:
    """
    Returns the given PauliArray broadcasted to a given shape.

    Args:
        paulis (PauliArray): PauliArray to broadcast.
        shape (Tuple[int, ...]): Shape to broadcast to.

    Returns:
        new_pauli_array (PauliArray): The PauliArray with a new shape.
    """

    new_shape = shape + (paulis.num_qubits,)

    new_z_strings = np.broadcast_to(paulis.z_strings, new_shape)
    new_x_strings = np.broadcast_to(paulis.x_strings, new_shape)

    return PauliArray(new_z_strings, new_x_strings)


def expand_dims(paulis: PauliArray, axis: Union[int, Tuple[int, ...]]) -> Self:
    """
    Expands the shape of a PauliArray.

    Inserts a new axis that will appear at the axis position in the expanded array shape.

    Args:
        paulis (PauliArray): The PauliArray to expand.
        axis (Union[int, Tuple[int, ...]]): The axis upon which expand the PauliArray.

    Returns:
        expanded_pauli_array (PauliArray) : The expanded PauliArray.
    """

    if type(axis) not in (tuple, list):
        axis = (axis,)

    actual_ndim = paulis.ndim
    for ax in axis:
        if ax > actual_ndim:
            raise ValueError("axis cannot exceed ndim")
        actual_ndim += 1

    new_z_strings = np.expand_dims(paulis.z_strings, axis)
    new_x_strings = np.expand_dims(paulis.x_strings, axis)

    return PauliArray(new_z_strings, new_x_strings)


def commutator(paulis_1: PauliArray, paulis_2: PauliArray) -> Tuple[PauliArray, "np.ndarray[np.complex]"]:
    """
    Returns the commutator of the two PauliArray parameters.

    Args:
        paulis_1 (PauliArray): PauliArray to calculate commmutator with.
        paulis_2 (PauliArray): Other PauliArray to calculate commmutator with.

    Returns:
        PauliArray: PauliArray containing the Pauli strings of the commutators.
        "np.ndarray[np.complex]" : Coefficients of Pauli strings in returned PauliArray.
    """
    assert is_broadcastable(paulis_1.shape, paulis_2.shape)

    commutators, phases = paulis_1.compose_pauli_array(paulis_2)
    do_commute = paulis_1.commute_with(paulis_2)

    commutators.z_strings[do_commute] = 0
    commutators.x_strings[do_commute] = 0
    phases[do_commute] = 0

    return commutators, phases


def commutator2(paulis_1: PauliArray, paulis_2: PauliArray) -> Tuple[PauliArray, "np.ndarray[np.complex]"]:
    """
    Returns the commutator of the two PauliArray parameters.

    Args:
        paulis_1 (PauliArray): PauliArray to calculate commmutator with.
        paulis_2 (PauliArray): Other PauliArray to calculate commmutator with.

    Returns:
        commutator_pauli_array (PauliArray): PauliArray containing the Pauli strings of the commutator.
        coefficients ("np.ndarray[np.complex]") : Coefficients of Pauli strings in returned PauliArray.
    """
    assert is_broadcastable(paulis_1.shape, paulis_2.shape)

    shape = broadcast_shape(paulis_1.shape, paulis_2.shape)

    do_commute = paulis_1.commute_with(paulis_2)

    idxs = np.where(~do_commute)

    idx1 = tuple(
        [idx if paulis_1.shape[dim] > 1 else np.zeros(idx.shape, dtype=np.int_) for dim, idx in enumerate(idxs)]
    )
    idx2 = tuple(
        [idx if paulis_2.shape[dim] > 1 else np.zeros(idx.shape, dtype=np.int_) for dim, idx in enumerate(idxs)]
    )

    non_zero_commutators, non_zeros_phases = paulis_1[*idx1].compose_pauli_array(paulis_2[*idx2])

    commutators = PauliArray.identities(shape, paulis_1.num_qubits)
    phases = np.zeros(shape, dtype=np.complex128)

    commutators[*idxs] = non_zero_commutators
    phases[*idxs] = non_zeros_phases

    return commutators, phases


def anticommutator(paulis_1: PauliArray, paulis_2: PauliArray) -> Tuple[PauliArray, "np.ndarray[np.complex]"]:
    """
    Returns the anticommutator of the two PauliArray parameters.

    Args:
        paulis_1 (PauliArray): PauliArray to calculate the anticommutator with.
        paulis_2 (PauliArray): Other PauliArray to calculate the anticommutator with.

    Returns:
        anticommutators_pauli_array (PauliArray): PauliArray containing the Pauli strings of the anticommutator.
        coefficients ("np.ndarray[np.complex]") : Coefficients of Pauli strings in returned PauliArray.
    """
    assert is_broadcastable(paulis_1.shape, paulis_2.shape)

    anticommutators, phases = paulis_1.compose_pauli_array(paulis_2)
    do_anticommute = ~paulis_1.commute_with(paulis_2)

    anticommutators.z_strings[do_anticommute] = 0
    anticommutators.x_strings[do_anticommute] = 0
    phases[do_anticommute] = 0

    return anticommutators, phases


def concatenate(paulis: Tuple[PauliArray, ...], axis: int) -> PauliArray:
    """
    Concatenated multiple PauliArrays.

    Args:
        paulis (List[PauliArray]): PauliArrays to concatenate.
        axis (int): The axis along which the arrays will be joined.

    Returns:
        PauliArray: The concatenated PauliArrays.
    """
    assert is_concatenatable(paulis, axis)

    z_strings_list = [pauli.z_strings for pauli in paulis]
    x_strings_list = [pauli.x_strings for pauli in paulis]
    new_z_strings = np.concatenate(z_strings_list, axis=axis)
    new_x_strings = np.concatenate(x_strings_list, axis=axis)

    return PauliArray(new_z_strings, new_x_strings)


def swapaxes(paulis: PauliArray, axis1: int, axis2: int) -> PauliArray:
    """
    Swap axes of a PauliArray

    Args:
        paulis (PauliArray): The PauliArray
        axis1 (int): Original axis position
        axis2 (int): Target axis position

    Returns:
        PauliArray: The PauliArrays with axes swaped.
    """
    assert axis1 < paulis.ndim
    assert axis2 < paulis.ndim
    axis1 = axis1 - 1 if axis1 < 0 else axis1
    axis2 = axis2 - 1 if axis2 < 0 else axis2

    new_z_strings = np.swapaxes(paulis.z_strings, axis1, axis2)
    new_x_strings = np.swapaxes(paulis.x_strings, axis1, axis2)

    return PauliArray(new_z_strings, new_x_strings)


def moveaxis(paulis: PauliArray, source: int, destination: int) -> PauliArray:
    """
    Move an axis of a PauliArray

    Args:
        paulis (PauliArray): The PauliArray
        source (int): Original axis position
        destination (int): Target axis position

    Returns:
        PauliArray: The PauliArrays with axis moved.
    """
    assert source < paulis.ndim
    assert destination < paulis.ndim
    source = source - 1 if source < 0 else source
    destination = destination - 1 if destination < 0 else destination

    new_z_strings = np.moveaxis(paulis.z_strings, source, destination)
    new_x_strings = np.moveaxis(paulis.x_strings, source, destination)

    return PauliArray(new_z_strings, new_x_strings)


def unique(
    paulis: PauliArray,
    axis: Optional[int] = None,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Union[PauliArray, Tuple[PauliArray, NDArray]]:
    """
    Finds unique elements in a PauliArray.
    Directly uses numpy.unique and has the same interface.

    Args:
        paulis (PauliArray): The PauliArray to return.

        axis (Optional[int], optional):  The axis to operate on. If None, the PauliArray will be flattened.
            If an integer, the subarrays indexed by the given axis will be flattened and treated as the elements
            of a 1-D array with the dimension of the given axis. Object arrays or structured arrays that contain
            objects are not supported if the axis kwarg is used. Defaults to None.

        return_index (bool, optional): If True, also return the indices of PauliArray (along the specified axis,
            if provided, or in the flattened array) that result in the unique array. Defaults to False.

        return_inverse (bool, optional): If True, also return the indices of the unique array
            (for the specified axis, if provided) that can be used to reconstruct array. Defaults to False.

        return_counts (bool, optional): If True, also return the number of times each unique item appears in array.
            Defaults to False.

    Returns:
        PauliArray: The unique Pauli strings (or PauliArray along an axis) in a PauliArray
        NDArray, optional: Index to get unique from the orginal PauliArray
        NDArray, optional: Innverse to reconstrut the original PauliArray from unique
        NDArray, optional: The number of each unique in the original PauliArray
    """

    if axis is None:
        paulis = paulis.flatten()
        axis = 0
    elif axis >= paulis.ndim:
        raise ValueError("")
    else:
        axis = axis % paulis.ndim

    out = np.unique(
        paulis.zx_strings,
        axis=axis,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if return_index or return_inverse or return_counts:
        out = list(out)
        unique_zx_strings = out[0]
        out[0] = PauliArray.from_zx_strings(unique_zx_strings)
    else:
        unique_zx_strings = out
        out = PauliArray.from_zx_strings(unique_zx_strings)

    return out


def fast_flat_unique(
    paulis: PauliArray,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Union[PauliArray, Tuple[PauliArray, NDArray]]:
    """
    Faster version of unique for PauliArray. Only works with flat PauliArray.
    Directly uses numpy.unique.

    Args:
        paulis (PauliArray): The PauliArray to return. Must be flat.

        return_index (bool, optional): If True, also return the indices of PauliArray (along the specified axis,
            if provided, or in the flattened array) that result in the unique array. Defaults to False.

        return_inverse (bool, optional): If True, also return the indices of the unique array
            (for the specified axis, if provided) that can be used to reconstruct array. Defaults to False.

        return_counts (bool, optional): If True, also return the number of times each unique item appears in array.
            Defaults to False.

    Returns:
        PauliArray: The unique Pauli strings in a PauliArray
        NDArray, optional: Index to get unique from the orginal PauliArray
        NDArray, optional: Innverse to reconstrut the original PauliArray from unique
        NDArray, optional: The number of each unique in the original PauliArray
    """

    assert paulis.ndim == 1

    zx_strings = paulis.zx_strings
    void_type_size = zx_strings.dtype.itemsize * 2 * paulis.num_qubits

    zx_view = np.squeeze(np.ascontiguousarray(zx_strings).view(np.dtype((np.void, void_type_size))), axis=-1)

    _, index, inverse, counts = np.unique(zx_view, return_index=True, return_inverse=True, return_counts=True)

    new_paulis = paulis[index]

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
