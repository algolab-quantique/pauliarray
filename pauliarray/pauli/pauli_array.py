from numbers import Number
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pauliarray.binary import bit_operations as bitops
from pauliarray.binary import symplectic
from pauliarray.binary import void_operations as vops
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

    def __init__(self, z_voids: NDArray[np.bool_], x_voids: NDArray[np.bool_], num_qubits: int):

        assert z_voids.shape == x_voids.shape
        assert z_voids.dtype.itemsize == x_voids.dtype.itemsize

        assert z_voids.dtype.itemsize * 8 - num_qubits >= 0
        # assert z_voids.dtype.itemsize * 8 - num_qubits <

        self._z_voids = z_voids
        self._x_voids = x_voids
        self._num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self._num_qubits

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the object.

        Returns:
            Tuple[int, ...]: The shape of the object.
        """
        return self._z_voids.shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions.

        Returns:
            int: The number of dimensions.
        """
        return self._z_voids.ndim

    @property
    def size(self) -> int:
        """
        Returns the total number of elements in the PauliArray.

        Returns:
            int: The total number of elements in the PauliArray.
        """
        return np.prod(self.shape)

    @property
    def paulis(self) -> "PauliArray":
        return self

    @property
    def x_voids(self) -> NDArray[np.bool_]:
        """
        Returns the X voids.

        Returns:
            NDArray[np.bool_]: The X bits.
        """
        return self._x_voids

    @property
    def z_voids(self) -> NDArray[np.bool_]:
        """
        Returns the Z voids.

        Returns:
            NDArray[np.bool_]: The Z bits.
        """
        return self._z_voids

    @property
    def zx_voids(self) -> NDArray[np.bool_]:
        """
        Returns the Z voids.

        Returns:
            NDArray[np.bool_]: The Z bits.
        """
        return vops.stich_voids(self._z_voids, self._x_voids)

    @property
    def x_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the X bits.

        Returns:
            "np.ndarray[np.bool]": The X bits.
        """
        return vops.voids_to_bit_strings(self._x_voids, self.num_qubits)

    @property
    def z_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the Z bits.

        Returns:
            "np.ndarray[np.bool]": The Z bits.
        """
        return vops.voids_to_bit_strings(self._z_voids, self.num_qubits)

    @property
    def zx_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the combined Z and X bits.

        Returns:
            "np.ndarray[np.bool]": The combined Z and X bits.
        """
        return symplectic.merge_zx_strings(self.z_strings, self.x_strings)

    @property
    def xz_strings(self) -> "np.ndarray[np.bool]":
        """
        Returns the combined X and Z bits.

        Returns:
            "np.ndarray[np.bool]": The combined X and Z bits.
        """
        return symplectic.merge_zx_strings(self.x_strings, self.z_strings)

    @property
    def num_ids(self) -> "np.ndarray[np.int]":
        """
        Returns the number of identity operators.

        Returns:
            "np.ndarray[np.int]": The number of identity operators.
        """

        return vops.bitwise_count(vops.paded_bitwise_not(vops.bitwise_or(self.z_voids, self.x_voids), self.num_qubits))

    @property
    def num_non_ids(self) -> "np.ndarray[np.int]":
        """
        Returns the number of non-identity operators.

        Returns:
            "np.ndarray[np.int]": The number of non-identity operators.
        """
        return vops.bitwise_count(vops.bitwise_or(self.z_voids, self.x_voids))

    def __getitem__(self, key):
        new_z_voids = self._z_voids[key]
        new_x_voids = self._x_voids[key]

        return PauliArray(new_z_voids, new_x_voids, self.num_qubits)

    def __setitem__(self, key, value: "PauliArray"):
        if isinstance(value, PauliArray):
            self._z_voids[key] = value.z_voids
            self._x_voids[key] = value.x_voids
        else:
            raise TypeError("Value should be of type PauliArray.")

    def __str__(self):
        return f"PauliArray: num_qubits = {self.num_qubits}, shape = {self.shape}, ..."

    def __eq__(self, other: "PauliArray") -> "np.ndarray[np.bool]":
        """
        Checks element-wise if the other PauliArray is equal.

        Args:
            other (PauliArray): An other PauliArray. Must be broadcastable

        Returns:
            "np.ndarray[np.bool]": _description_
        """

        return np.logical_and((self.z_voids == other.z_voids), (self.x_voids == other.x_voids))

    def copy(self) -> "PauliArray":
        """
        Returns a copy of the PauliArray.

        Returns:
            PauliArray: Copied PauliArray.
        """
        return PauliArray(self.z_voids.copy(), self.x_voids.copy(), self.num_qubits)

    def reshape(self, shape: Tuple[int, ...]) -> "PauliArray":
        """
        Returns a PauliArray with a new shape.

        Args:
            shape (tuple[int]): Tuple containing the new shape e.g. for 2x2 matrix: shape = (2,2)

        Returns:
            PauliArray: The PauliArray object with the new shape.
        """

        new_z_voids = self._z_voids.reshape(shape)
        new_x_voids = self._x_voids.reshape(shape)

        return PauliArray(new_z_voids, new_x_voids, self.num_qubits)

    def flatten(self) -> "PauliArray":
        """
        Returns a copy of the PauliArray flattened into one dimension.

        Returns:
            PauliArray: A flattened copy of the current PauliArray.
        """

        shape = (np.prod(self.shape),)

        return self.reshape(shape)

    def squeeze(self) -> "PauliArray":
        """
        Returns a PauliArray with axes of length one removed.

        Returns:
            PauliArray: The squeezed PauliArray.
        """

        new_z_voids = self._z_voids.squeeze()
        new_x_voids = self._x_voids.squeeze()

        return PauliArray(new_z_voids, new_x_voids, self.num_qubits)

    def remove(self, index: int) -> "PauliArray":
        """
        Returns a PauliArray with removed item at given index.

        Args:
            index (int): Index of element to remove.

        Returns:
            PauliArray: PauliArray with removed item at given index.
        """

        new_z_voids = self._z_voids.remove(index)
        new_x_voids = self._x_voids.remove(index)

        return PauliArray(new_z_voids, new_x_voids, self.num_qubits)

    def take_qubits(self, indices: Union["np.ndarray[np.int]", range, int], inplace: bool = True) -> "PauliArray":
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

        return PauliArray.from_z_strings_and_x_strings(new_z_strings, new_x_strings)

    def compress_qubits(self, condition: "np.ndarray[np.bool]", inplace: bool = True) -> "PauliArray":
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

        return PauliArray.from_z_strings_and_x_strings(new_z_strings, new_x_strings)

    def reorder_qubits(self, qubit_order: List[int], inplace: bool = True) -> "PauliArray":
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

        new_z_strings = self.z_strings[..., qubit_order]
        new_x_strings = self.x_strings[..., qubit_order]

        return PauliArray.from_z_strings_and_x_strings(new_z_strings, new_x_strings)

    def compose(self, other: Any) -> Any:

        if isinstance(other, PauliArray):
            return self.compose_pauli_array(other)

        return NotImplemented

    def compose_pauli_array(self, other: "PauliArray") -> Tuple["PauliArray", "np.ndarray[np.complex]"]:
        """
        Performs an element-wise composition with an other PauliArray.

        Args:
            other (PauliArray): The PauliArray to compose with. Must be broadcastable.

        Returns:
            PauliArray: The result of the composition.
            "np.ndarray[np.complex]" : Phases resulting from the composition.
        """
        assert self.num_qubits == other.num_qubits
        assert is_broadcastable(self.shape, other.shape)

        new_z_voids = vops.bitwise_xor(self.z_voids, other.z_voids)
        new_x_voids = vops.bitwise_xor(self.x_voids, other.x_voids)

        self_phase_power = vops.bitwise_dot(self.z_voids, self.x_voids)
        other_phase_power = vops.bitwise_dot(other.z_voids, other.x_voids)
        new_phase_power = vops.bitwise_dot(new_z_voids, new_x_voids)
        commutation_phase_power = 2 * vops.bitwise_dot(self.x_voids, other.z_voids)

        phase_power = commutation_phase_power + self_phase_power + other_phase_power - new_phase_power

        phases = np.choose(phase_power, [1, -1j, -1, 1j], mode="wrap")

        return PauliArray(new_z_voids, new_x_voids, self.num_qubits), phases

    def mul_weights(self, other: Union[Number, NDArray]) -> "WeightedPauliArray":
        """
        Apply a weight to each Pauli string to form a WeightedPauliArray

        Args:
            other (Union[Number, NDArray]): A number or an array of number. Must be broadcastable.

        Returns:
            WeightedPauliArray: The result of the weight application.
        """

        from pauliarray.pauli.weighted_pauli_array import WeightedPauliArray

        new_weights = np.broadcast_to(other, self.shape).astype(np.complex128)
        new_paulis = self.paulis.copy()

        return WeightedPauliArray(new_paulis, new_weights)

    def tensor(self, other: Any) -> Any:

        if isinstance(other, PauliArray):
            return self.tensor_pauli_array(other)

        return NotImplemented

    def tensor_pauli_array(self, other: "PauliArray") -> "PauliArray":
        """
        Performs a tensor product, element-wise with an other PauliArray.

        Args:
            other (PauliArray): The PauliArray to multiply with. Must be broadcastable.

        Returns:
            PauliArray: The result of the tensor product.
        """
        new_z_strings = np.concatenate((self.z_strings, other.z_strings), axis=-1)
        new_x_strings = np.concatenate((self.x_strings, other.x_strings), axis=-1)

        return PauliArray.from_z_strings_and_x_strings(new_z_strings, new_x_strings)

    def add_pauli_array(self, other: "PauliArray") -> "OperatorArrayType1":
        """
        Performs an element-wise addition with other Pauli Array to produce an array of operator.

        Args:
            other (PauliArray): The PauliArray to add. Must be broadcastable.

        Returns:
            OperatorArrayType1: The result of the addition as an array of operators.
        """

        from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1

        assert self.num_qubits == other.num_qubits
        assert is_broadcastable(self.shape, other.shape)

        new_z_strings = np.stack((self.z_strings, other.z_strings), axis=-2)
        new_x_strings = np.stack((self.x_strings, other.x_strings), axis=-2)

        new_paulis = PauliArray.from_z_strings_and_x_strings(new_z_strings, new_x_strings)

        return OperatorArrayType1.from_pauli_array(new_paulis, -1)

    def sum(self, axis: Union[Tuple[int, ...], None] = None) -> "OperatorArrayType1":

        from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1

        OperatorArrayType1.from_pauli_array(self)

    def flip_zx(self) -> "PauliArray":
        """
        Returns a copy of the PauliArray with x strings as z strings, and vice versa.

        Returns:
            PauliArray: The PauliArray with flipped z and x strings.
        """
        return PauliArray(self.x_voids.copy(), self.z_voids.copy(), self.num_qubits)

    def commute_with(self, other: "PauliArray") -> "np.ndarray[np.bool]":
        """
        Returns True if the elements of PauliArray commutes with the elements of PauliArray passed as parameter,
        returns False otherwise.

        Args:
            other (PauliArray): The PauliArray to check commutation with.

        Returns:
            "np.ndarray[np.bool]": An array of bool set to true for commuting Pauli string, and false otherwise.
        """

        return ~np.mod(
            vops.bitwise_dot(self.z_voids, other.x_voids) + vops.bitwise_dot(self.x_voids, other.z_voids), 2
        ).astype(np.bool_)

    def bitwise_commute_with(self, other: "PauliArray") -> "np.ndarray[np.bool]":
        """
        Returns True if the elements of PauliArray commutes bitwise with the elements of
        PauliArray passed as parameter, returns False otherwise.

        Args:
            other (PauliArray): The other PauliArray to verify bitwise commutation with.

        Returns:
            "np.ndarray[np.bool]": An array of bool set to true for bitwise commuting Pauli string, and false otherwise.
        """

        ovlp_1 = vops.bitwise_and(self.z_voids, other.x_voids)
        ovlp_2 = vops.bitwise_and(self.x_voids, other.z_voids)

        olvp_3 = vops.bitwise_xor(ovlp_1, ovlp_2)

        return olvp_3 == vops.bit_strings_to_voids(np.zeros(self.num_qubits, dtype=np.uint8))

    def traces(self) -> NDArray:
        """
        Return the traces of the Pauli Strings which are 2^n if Identity and 0 otherwise.

        Returns:
            "np.ndarray[np.int]": Traces of the Pauli Strings
        """

        return 2**self.num_qubits * (self.num_ids == self.num_qubits)

    def generators(self) -> "PauliArray":
        """
        Finds a set of linearly independant PauliString which can be multiplied together to generate every PauliStirng
        in self.

        Returns:
            PauliArray: The generators
        """

        generator_zx_strings = bitops.row_space(self.flatten().zx_strings)

        generators = PauliArray.from_z_strings_and_x_strings(
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
            label_table = self.label_table_1d(self.to_labels())
            return f"PauliArray\n{label_table}"

        if self.ndim == 2:
            label_table = self.label_table_2d(self.to_labels())
            return f"PauliArray\n{label_table}"

        label_table = self.label_table_nd(self.to_labels())
        return f"PauliArray\n{label_table}"

    def x(self, qubits: Union[int, List[int]], inplace: bool = True) -> Tuple["PauliArray", "np.ndarray[np.complex]"]:
        """
        Performs a Clifford conjugaison by X on given qubits. This leaves the PauliStrings unchanged but produce phase
        factors -1 when an operator is Y or Z.

        Args:
            qubits (int or list[int]): The qubits on which to apply the X
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        if not inplace:
            return self.copy().x(qubits)

        qubit_mask = np.zeros((self.num_qubits,), dtype=bool)
        qubit_mask[qubits] = True
        qubit_void_mask = vops.bit_strings_to_voids(qubit_mask)

        y_voids = vops.bitwise_and(self.x_voids, self.z_voids)
        sign_phases = np.mod(
            vops.bitwise_count(vops.bitwise_and(vops.bitwise_or(y_voids, self.z_voids), qubit_void_mask)), 2
        )

        factors = (1 - 2 * sign_phases).astype(complex)

        return self, factors

    def h(self, qubits: Union[int, List[int]], inplace: bool = True) -> Tuple["PauliArray", "np.ndarray[np.complex]"]:
        """
        Performs a Clifford conjugaison by H on given qubits. This exchanges X for Z and vice-versa and Y into -Y.

        Args:
            qubits (int or list[int]): The qubits on which to apply H.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        if not inplace:
            return self.copy().h(qubits)

        qubit_mask = np.zeros((self.num_qubits,), dtype=bool)
        qubit_mask[qubits] = True
        qubit_void_mask = vops.bit_strings_to_voids(qubit_mask)
        qubit_void_not_mask = vops.bit_strings_to_voids(~qubit_mask)

        y_voids = vops.bitwise_and(self.x_voids, self.z_voids)
        sign_phases = np.mod(vops.bitwise_count(vops.bitwise_and(y_voids, qubit_void_mask)), 2)

        self._z_voids, self._x_voids = vops.bitwise_or(
            vops.bitwise_and(qubit_void_not_mask, self.z_voids), vops.bitwise_and(qubit_void_mask, self.x_voids)
        ), vops.bitwise_or(
            vops.bitwise_and(qubit_void_not_mask, self.x_voids), vops.bitwise_and(qubit_void_mask, self.z_voids)
        )

        factors = (1 - 2 * sign_phases).astype(complex)

        return self, factors

    def s(self, qubits: Union[int, List[int]], inplace: bool = True) -> Tuple["PauliArray", "np.ndarray[np.complex]"]:
        """
        Performs a Clifford conjugaison by S on given qubits. This exchanges X for Y and vice-versa with respective factors.

        Args:
            qubits (int or list[int]): The qubits on which to apply S.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        if not inplace:
            return self.copy().s(qubits)

        qubit_mask = np.zeros((self.num_qubits,), dtype=bool)
        qubit_mask[qubits] = True
        qubit_void_mask = vops.bit_strings_to_voids(qubit_mask)
        qubit_void_not_mask = vops.bit_strings_to_voids(~qubit_mask)

        y_voids = vops.bitwise_and(self.x_voids, self.z_voids)
        sign_phases = np.mod(vops.bitwise_count(vops.bitwise_and(y_voids, qubit_void_mask)), 2)
        factors = (1 - 2 * sign_phases).astype(complex)

        self._z_voids = vops.bitwise_or(
            vops.bitwise_and(qubit_void_not_mask, self.z_voids),
            vops.bitwise_and(qubit_void_mask, vops.bitwise_xor(self.z_voids, self.x_voids)),
        )

        return self, factors

    def cx(
        self, control_qubits: Union[int, List[int]], target_qubits: Union[int, List[int]], inplace: bool = True
    ) -> Tuple["PauliArray", "np.ndarray[np.complex]"]:
        """
        Performs a Clifford conjugaison by CX on given qubits. The order of the CX is set by the order of the qubits.

        Args:
            control_qubits (int or list[int]): The qubits which controls the CX.
            target_qubits (int or list[int]): The qubits target by CX.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        if not inplace:
            return self.copy().cx(control_qubits, target_qubits)

        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        assert len(control_qubits) == len(target_qubits)

        factors = np.ones(self.shape, dtype=complex)
        x_strings = self.x_strings
        z_strings = self.z_strings
        for cq, tq in zip(control_qubits, target_qubits):

            factors *= 1 - 2 * x_strings[..., cq] * z_strings[..., tq] * np.logical_not(
                np.logical_xor(z_strings[..., cq], x_strings[..., tq])
            )

            tmp_tq_x_bit_array = x_strings[..., tq].copy()
            tmp_cq_z_bit_array = z_strings[..., cq].copy()
            x_strings[..., tq] = np.logical_xor(tmp_tq_x_bit_array, x_strings[..., cq])
            z_strings[..., cq] = np.logical_xor(tmp_cq_z_bit_array, z_strings[..., tq])

        self._z_voids = vops.bit_strings_to_voids(z_strings)
        self._x_voids = vops.bit_strings_to_voids(x_strings)

        return self, factors

    def cz(
        self, control_qubits: Union[int, List[int]], target_qubits: Union[int, List[int]], inplace: bool = True
    ) -> Tuple["PauliArray", "np.ndarray[np.complex]"]:
        """
        Performs a Clifford conjugaison by CZ on given qubits. The order of the CZ is set by the order of the qubits.

        Args:
            control_qubits (int or list[int]): The qubits which controls the CZ.
            target_qubits (int or list[int]): The qubits target by CZ.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        if not inplace:
            return self.copy().cz(control_qubits, target_qubits)

        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        assert len(control_qubits) == len(target_qubits)

        factors = np.ones(self.shape, dtype=complex)
        x_strings = self.x_strings
        z_strings = self.z_strings
        for cq, tq in zip(control_qubits, target_qubits):
            factors *= 1 - 2 * x_strings[..., cq] * x_strings[..., tq] * np.logical_xor(
                z_strings[..., cq], z_strings[..., tq]
            )

            z_strings[..., cq] = np.logical_xor(z_strings[..., cq], x_strings[..., tq])
            z_strings[..., tq] = np.logical_xor(z_strings[..., tq], x_strings[..., cq])

        self._z_voids = vops.bit_strings_to_voids(z_strings)
        self._x_voids = vops.bit_strings_to_voids(x_strings)

        return self, factors

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
        # assert self.num_qubits < 30

        mat_shape = (2**self.num_qubits, 2**self.num_qubits)
        matrices = np.zeros(self.shape + mat_shape, dtype=complex)

        z_ints = vops.voids_to_int_strings(self.z_voids)
        x_ints = vops.voids_to_int_strings(self.x_voids)

        phase_powers = np.mod(vops.bitwise_dot(self.z_voids, self.x_voids).astype(np.int32), 4)
        phases = np.choose(phase_powers, [1, -1j, -1, 1j])

        for idx in np.ndindex(self.shape):
            one_matrix = self.matrix_from_zx_ints(z_ints[idx], x_ints[idx], self.num_qubits)
            matrices[idx] = one_matrix

        return phases[..., None, None] * matrices

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

        row_ind = np.arange(dim, dtype=np.uint64)
        col_ind = np.bitwise_xor(row_ind, x_int)

        matrix_elements = 1 - 2 * (np.bitwise_count(np.bitwise_and(row_ind, z_int)).astype(np.float64) % 2)

        return row_ind, col_ind, matrix_elements

    @classmethod
    def from_labels(cls, labels: Union[list[str], "np.ndarray[np.str]"]) -> "PauliArray":
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

        return PauliArray.from_z_strings_and_x_strings(z_strings, x_strings)

    @classmethod
    def from_zx_voids(cls, zx_voids: NDArray, num_qubits: int) -> "PauliArray":

        return PauliArray(*vops.split_voids(zx_voids), num_qubits)

    @classmethod
    def from_z_strings_and_x_strings(cls, z_strings: NDArray[np.bool_], x_strings: NDArray[np.bool_]) -> "PauliArray":
        """
        Create a PauliArray from zx strings.

        Args:
            zx_strings (NDArray[np.bool_]): Array where the last dimension size is an even integers (twice the number of qubits.)

        Returns:
            PauliArray: The created PauliArray .
        """

        z_voids = vops.bit_strings_to_voids(z_strings)
        x_voids = vops.bit_strings_to_voids(x_strings)
        num_qubits = z_strings.shape[-1]

        return PauliArray(z_voids, x_voids, num_qubits)

    @classmethod
    def from_zx_strings(cls, zx_strings: "np.ndarray[np.bool]") -> "PauliArray":
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

        return PauliArray.from_z_strings_and_x_strings(z_strings, x_strings)

    @classmethod
    def identities(cls, shape: Tuple[int, ...], num_qubits: int) -> "PauliArray":
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

        return PauliArray.from_z_strings_and_x_strings(z_strings, x_strings)

    @classmethod
    def new(cls, shape: Tuple[int, ...], num_qubits: int) -> "PauliArray":

        return cls.identities(shape, num_qubits)

    @classmethod
    def random(cls, shape: Tuple[int, ...], num_qubits: int) -> "PauliArray":
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

        return PauliArray.from_z_strings_and_x_strings(z_strings, x_strings)

    @staticmethod
    def labels_to_z_strings_x_strings(
        labels: Union[list[str], "np.ndarray[np.str]"]
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

    @staticmethod
    def label_table_1d(labels) -> str:

        return "\n".join(labels)

    @staticmethod
    def label_table_2d(labels) -> str:

        row_strs = []
        for i in range(labels.shape[0]):
            row_strs.append("  ".join(labels[i, :]))

        return "\n".join(row_strs)

    @staticmethod
    def label_table_nd(labels) -> str:

        slice_strs = []
        for idx in np.ndindex(labels.shape[:-2]):
            slice_str = "Slice (" + ",".join([str(i) for i in idx]) + ",:,:)\n"
            slice_str += PauliArray.label_table_2d(labels[idx])
            slice_strs.append(slice_str)

        return "\n".join(slice_strs)


def argsort(paulis: PauliArray, axis: int = -1) -> "np.ndarray[np.int]":
    """
    Return indices which sorts the Pauli Strings.

    Returns:
        NDArray: Indices which sorts the Pauli Strings.
    """
    zx_ints = bitops.strings_to_ints(paulis.zx_strings)

    return np.argsort(zx_ints, axis)


def broadcast_to(paulis: PauliArray, shape: Tuple[int, ...]) -> "PauliArray":
    """
    Returns the given PauliArray broadcasted to a given shape.

    Args:
        paulis (PauliArray): PauliArray to broadcast.
        shape (Tuple[int, ...]): Shape to broadcast to.

    Returns:
        new_pauli_array (PauliArray): The PauliArray with a new shape.
    """

    new_z_voids = np.broadcast_to(paulis.z_voids, shape)
    new_x_voids = np.broadcast_to(paulis.x_voids, shape)

    return PauliArray(new_z_voids, new_x_voids, paulis.num_qubits)


def expand_dims(paulis: PauliArray, axis: Union[int, Tuple[int, ...]]) -> "PauliArray":
    """
    Expands the shape of a PauliArray.

    Inserts a new axis that will appear at the axis position in the expanded array shape.

    Args:
        paulis (PauliArray): The PauliArray to expand.
        axis (Union[int, Tuple[int, ...]]): The axis upon which expand the PauliArray.

    Returns:
        expanded_pauli_array (PauliArray) : The expanded PauliArray.
    """

    new_z_voids = np.expand_dims(paulis.z_voids, axis)
    new_x_voids = np.expand_dims(paulis.x_voids, axis)

    return PauliArray(new_z_voids, new_x_voids, paulis.num_qubits)


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

    zero_void = vops.bit_strings_to_voids(np.zeros(2 * paulis_1.num_qubits, dtype=np.uint8))

    commutators.z_voids[do_commute] = zero_void
    commutators.x_voids[do_commute] = zero_void

    coefs = 2 * phases * ~do_commute

    return commutators, coefs


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

    non_zero_commutators, non_zeros_coefs = paulis_1[*idx1].compose_pauli_array(paulis_2[*idx2])

    commutators = PauliArray.identities(shape, paulis_1.num_qubits)
    coefs = np.zeros(shape, dtype=np.complex128)

    commutators[*idxs] = non_zero_commutators
    coefs[*idxs] = 2 * non_zeros_coefs

    return commutators, coefs


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
    do_commute = paulis_1.commute_with(paulis_2)

    zero_void = vops.bit_strings_to_voids(np.zeros(2 * paulis_1.num_qubits, dtype=np.uint8))

    anticommutators.z_voids[~do_commute] = zero_void
    anticommutators.x_voids[~do_commute] = zero_void

    coefs = 2 * phases * do_commute

    return anticommutators, coefs


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

    z_voids_list = [pauli.z_voids for pauli in paulis]
    x_voids_list = [pauli.x_voids for pauli in paulis]
    new_z_voids = np.concatenate(z_voids_list, axis=axis)
    new_x_voids = np.concatenate(x_voids_list, axis=axis)

    return PauliArray(new_z_voids, new_x_voids, paulis[0].num_qubits)


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

    new_z_voids = np.swapaxes(paulis.z_voids, axis1, axis2)
    new_x_voids = np.swapaxes(paulis.x_voids, axis1, axis2)

    return PauliArray(new_z_voids, new_x_voids, paulis.num_qubits)


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

    new_z_voids = np.moveaxis(paulis.z_voids, source, destination)
    new_x_voids = np.moveaxis(paulis.x_voids, source, destination)

    return PauliArray(new_z_voids, new_x_voids, paulis.num_qubits)


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
        paulis.zx_voids,
        axis=axis,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if return_index or return_inverse or return_counts:
        out = list(out)
        unique_zx_voids = out[0]
        out[0] = PauliArray.from_zx_voids(unique_zx_voids, paulis.num_qubits)
    else:
        unique_zx_voids = out
        out = PauliArray.from_zx_voids(unique_zx_voids, paulis.num_qubits)

    return out
