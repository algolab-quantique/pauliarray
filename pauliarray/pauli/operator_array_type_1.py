from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.utils.array_operations import broadcast_shape, broadcasted_index, is_broadcastable, is_concatenatable

if TYPE_CHECKING:
    from pauliarray.pauli.operator import Operator


class OperatorArrayType1(object):
    """
    Defines an OperatorArray that contains an array of Operator. Each op.Operator has the same number of Pauli strings.
    Based on the wpa.WeightedPauliArray where the last dimension is used for summation.
    """

    def __init__(self, wpaulis: wpa.WeightedPauliArray):
        """
        Initializes the OperatorArrayType1 with a WeightedPauliArray.

        Args:
            wpaulis (wpa.WeightedPauliArray): The WeightedPauliArray to initialize with.
        """
        self._wpaulis = wpaulis

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the operator array."""
        return self._wpaulis.num_qubits

    @property
    def num_terms(self) -> int:
        """Number of terms in the operator array."""
        return self._wpaulis.shape[-1]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the operator array, excluding the last dimension."""
        return self._wpaulis.shape[:-1]

    @property
    def ndim(self) -> int:
        """Number of dimensions in the operator array, excluding the last dimension."""
        return self._wpaulis.ndim - 1

    @property
    def size(self):
        """Total number of elements in the operator array."""
        return np.prod(self.shape)

    @property
    def wpaulis(self) -> wpa.WeightedPauliArray:
        """WeightedPauliArray object representing the current operator array."""
        return self._wpaulis

    @property
    def paulis(self) -> pa.PauliArray:
        """PauliArray object representing the current operator array."""
        return self._wpaulis.paulis

    @property
    def weights(self) -> NDArray[np.complex_]:
        """Weights of the Pauli terms in the operator array."""
        return self._wpaulis.weights

    def __str__(self) -> str:
        """String representation of current operator array"""
        return f"OperatorUniformArray: num_qubits = {self.num_qubits}, num_terms = {self.num_terms}, ..."

    def __getitem__(self, key) -> Self:
        """
        Gets an item from the operator array.

        Args:
            key: The index or slice to access the operator array.

        Returns:
            OperatorArrayType1: A new OperatorArrayType1 instance with the accessed item.
        """
        return OperatorArrayType1(self._wpaulis[key])

    def __eq__(self, other: Self) -> NDArray[np.bool_]:
        """
        Checks element-wise if the operators in the array are equal to the other.

        Args:
            other (OperatorArrayType1): Another OperatorArrayType1. Must be broadcastable.

        Returns:
            NDArray[np.bool_]: An array indicating where the operators are equal.
        """
        new_shape = broadcast_shape(self.shape, other.shape)

        eq_array = np.empty(new_shape, dtype=np.bool_)
        for idx in np.ndindex(new_shape):
            idx1 = broadcasted_index(self.shape, idx)
            idx2 = broadcasted_index(other.shape, idx)

            eq_array[idx] = self.get_operator(*idx1) == other.get_operator(*idx2)

        return eq_array

    def _mul(self, other: ArrayLike) -> Self:
        """
        Multiplies the operator array with another array-like object.

        Args:
            other (ArrayLike): The array-like object to multiply with.

        Returns:
            OperatorArrayType1: A new OperatorArrayType1 instance with the result.
        """
        return OperatorArrayType1(self.wpaulis * other)

    __mul__ = __rmul__ = _mul

    def copy(self) -> Self:
        """
        Creates a copy of the operator array.

        Returns:
            OperatorArrayType1: A copy of the operator array.
        """
        return OperatorArrayType1(self._wpaulis.copy())

    def adjoint(self) -> Self:
        """
        Computes the adjoint of the operator array.

        Returns:
            OperatorArrayType1: The adjoint of the operator array.
        """
        return OperatorArrayType1(self.wpaulis.adjoint())

    def reshape(self, shape: Tuple[int, ...]) -> Self:
        """
        Reshapes the operator array.

        Args:
            shape (Tuple[int, ...]): New shape.

        Returns:
            OperatorArrayType1: Reshaped operator array.
        """
        new_wpaulis_shape = shape + (self.num_terms,)

        return OperatorArrayType1(self.wpaulis.reshape(new_wpaulis_shape))

    def flatten(self) -> Self:
        """
        Flattens the operator array into one dimension.

        Returns:
            OperatorArrayType1: A flattened copy of the operator array.
        """
        shape = (np.prod(self.shape, dtype=int),)

        return self.reshape(shape)

    def compose(self, other: Any) -> Any:
        """
        Composes the operator array with another operator array or object.
        Supports : OperatorArrayType1. Exists for coherence between all PauliArray data structures and
        raises NotImplemented if other type is inputed.

        Args:
            other (Any): The other object to compose with.

        Returns:
            Any: The resulting OperatorArrayType1 or NotImplemented Error.
        """
        if isinstance(other, OperatorArrayType1):
            return self.compose_operator_array_type_1(other)

        return NotImplemented

    def compose_operator_array_type_1(self, other: Self) -> Self:
        """
        Composes the operator array with another OperatorArrayType1. Both objects must be broadcastable following
        numpy's broadcasting rules.

        Args:
            other (OperatorArrayType1): The other OperatorArrayType1 to compose with.

        Returns:
            OperatorArrayType1: The composition result.
        """
        assert self.ndim == other.ndim
        assert is_broadcastable(self.shape, other.shape)

        exp_self_wpaulis = wpa.expand_dims(self.wpaulis, axis=self.ndim + 1)
        exp_other_wpaulis = wpa.expand_dims(other.wpaulis, axis=other.ndim)

        exp_prod_wpaulis = exp_self_wpaulis.compose_weighted_pauli_array(exp_other_wpaulis)

        final_shape = np.broadcast_shapes(self.shape, other.shape) + (self.num_terms * other.num_terms,)

        return OperatorArrayType1(exp_prod_wpaulis.reshape(final_shape))

    def mul_weights(self, other: NDArray) -> Self:
        """
        Multiplies the weights of the operator array with another array of weights. Both objects must be
        broadcastable following numpy's broadcasting rules.

        Args:
            other (NDArray): The array to multiply weights with.

        Returns:
            OperatorArrayType1: The resulting operator array with multiplied weights.
        """
        other = np.array(other)
        assert is_broadcastable(self.shape, other.shape)

        exp_other = np.expand_dims(other, other.ndim)
        new_wpaulis = self.wpaulis.mul_weights(exp_other)

        return OperatorArrayType1(new_wpaulis)

    def add(self, other: Any) -> Any:
        """
        Add another operator array to the current operator array.
        Supports : OperatorArrayType1. Exists for coherence between all PauliArray data structures and
        raises NotImplemented if other type is inputed.

        Args:
            other (Any): The other object to add.

        Returns:
            Any: The resulting OperatorArrayType1 or NotImplemented Error.
        """
        if isinstance(other, OperatorArrayType1):
            return self.add_operator_array_type_1(other)

        return NotImplemented

    def add_operator_array_type_1(self, other: Self) -> Self:
        """
        Add another OperatorArrayType1 to the current operator array.

        Args:
            other (OperatorArrayType1): The other OperatorArrayType1 to add.

        Returns:
            OperatorArrayType1: The resulting OperatorArrayType1.
        """
        new_wpaulis = wpa.concatenate([self.wpaulis, other.wpaulis], self.ndim)

        return OperatorArrayType1(new_wpaulis)

    def sum(self, axes: Union[Tuple[int, ...], None] = None) -> Union[op.Operator, "OperatorArrayType1"]:
        """
        Carry out summation along the given axes. Effectively, combining these axes with the hidden summation axis. Returns an Operator if a single operator remains in the operator array.

        Returns:
            _type_: _description_
        """

        if axes is None:
            return op.Operator(self.wpaulis.flatten())

        if isinstance(axes, int):
            axes = (axes,)

        axes = [axis - 1 if axis < 0 else axis for axis in axes]

        num_sum_axis = len(axes) + 1
        num_keep_axis = self.wpaulis.ndim - num_sum_axis

        axes = sorted(axes)[::-1]

        new_wpaulis = self.wpaulis.copy()
        for i, ax in enumerate(axes):
            new_wpaulis = wpa.moveaxis(new_wpaulis, ax, -1 - i)

        new_wshape = new_wpaulis.shape[:num_keep_axis] + (np.prod(new_wpaulis.shape[num_keep_axis:]),)
        new_wpaulis = new_wpaulis.reshape(new_wshape)

        if new_wpaulis.ndim == 1:
            return op.Operator(new_wpaulis)

        return OperatorArrayType1(new_wpaulis)

    def add_scalar(self, other: NDArray) -> Self:
        """
        Add scalar array to the operator array.

        Args:
            other (NDArray): The scalar array to add.

        Returns:
            OperatorArrayType1: The resulting operator array.
        """
        other_operators = OperatorArrayType1.from_pauli_array(
            pa.PauliArray.identities(self.shape, self.num_qubits)
        ).mul_weights(other)

        return self.add_operator_array_type_1(other_operators)

    def get_operator(self, *idx) -> op.Operator:
        """
        Returns a single operator in operator array.

        Args:
            idx: Indices to access the operator.

        Returns:
            op.Operator: The accessed operator.
        """
        assert len(idx) == self.ndim

        return op.Operator(self.wpaulis[idx])

    def inspect(self) -> str:
        """
        Creates a string describing the operator array.

        Returns:
            str: String representation of the operator array details.
        """
        detail_str = "OperatorArray"

        for idx in np.ndindex(self.shape):
            operator = self._wpaulis[idx]
            detail_str += f"\n ---{idx}--- " + operator.inspect()

        return detail_str

    def x(self, qubits: Union[int, List[int]], inplace: bool = True) -> "OperatorArrayType1":
        """
        Apply X transformations on qubits of Operators. This leaves the PauliStrings unchanged but produce
        phase factors -1 when operators are Y or Z.

        Args:
            qubits (int or list[int]): The qubits on which to apply the X.
            inplace (bool): Apply the changes to self if True. Return a modified copy if False.

        Returns:
            self_copy (WeightedPauliArray): A modified copy of self, only if inplace=True
        """

        if not inplace:
            return self.copy().x(qubits)

        self.wpaulis.x(qubits, inplace=True)
        return self

    def y(self, qubits: Union[int, List[int]]) -> Self:
        """
        Applies a Y gate to the specified qubits.

        Args:
            qubits (Union[int, List[int]]): The qubits to apply the Y gate to.

        Returns:
            OperatorArrayType1: The resulting operator array.
        """
        return self.apply(op.Operator.y, qubits)

    def z(self, qubits: Union[int, List[int]]) -> Self:
        """
        Applies a Z gate to the specified qubits.

        Args:
            qubits (Union[int, List[int]]): The qubits to apply the Z gate to.

        Returns:
            OperatorArrayType1: The resulting operator array.
        """
        return self.apply(op.Operator.z, qubits)

    def h(self, qubits: Union[int, List[int]]) -> Self:
        """
        Applies a Hadamard gate to the specified qubits.

        Args:
            qubits (Union[int, List[int]]): The qubits to apply the Hadamard gate to.

        Returns:
            OperatorArrayType1: The resulting operator array.
        """
        return self.apply(op.Operator.h, qubits)

    def apply(self, func: Callable, qubits: Union[int, List[int]]) -> Self:
        """
        Applies a function (operation) to the specified qubits.

        Args:
            func (Callable): The opreation to apply.
            qubits (Union[int, List[int]]): The qubits to apply the operation to.

        Returns:
            OperatorArrayType1: The resulting operator array.
        """
        qubits = [qubits] if isinstance(qubits, int) else qubits

        new_wpaulis = self.wpaulis.apply(func, qubits)

        return OperatorArrayType1(new_wpaulis)

    def drop_identity(self) -> Self:
        """
        Drops the identity from the operator array.

        Returns:
            OperatorArrayType1: The resulting operator array without the identity component.
        """
        return OperatorArrayType1(self._wpaulis.drop_identity())

    def remove_small_weights(self, threshold: float = 1e-14) -> "OperatorArrayType1":
        return self.filter_weights(lambda weight: np.abs(weight) > threshold)

    @classmethod
    def from_operator(cls, opers: op.Operator) -> Self:
        """
        Creates an OperatorArrayType1 from Operator.

        Args:
            opers (Operator): The operator to create from.

        Returns:
            OperatorArrayType1: The resulting operator array.
        """
        return cls(wpa.WeightedPauliArray.from_operator(opers))

    @classmethod
    def from_pauli_array(
        cls, paulis: pa.PauliArray, summation_axis: Union[Tuple[int, ...], None] = None
    ) -> "OperatorArrayType1":
        """
        Converts a PauliArray into an OperatorArrayType1.

        Args:
            paulis (pa.PauliArray): The PauliArray
            summation_axis (int, optional): Which axis of PauliArray to use for summation. If None, each Pauli string becomes an operator. Defaults to None.

        Returns:
            OperatorArrayType1: _description_
        """

        new_paulis = paulis.reshape(paulis.shape + (1,))

        new_operators = cls(wpa.WeightedPauliArray.from_paulis(new_paulis))

        if summation_axis is not None:
            new_operators = new_operators.sum(summation_axis)

        return new_operators

    @classmethod
    def from_weighted_pauli_array(
        cls, wpaulis: wpa.WeightedPauliArray, summation_axis: Union[Tuple[int, ...], None] = None
    ) -> "OperatorArrayType1":

        new_wpaulis = wpaulis.reshape(wpaulis.shape + (1,))

        new_operators = cls(new_wpaulis)

        if summation_axis is not None:
            new_operators = new_operators.sum(summation_axis)

        return new_operators

    @classmethod
    def from_operator_list(cls, operators: List[op.Operator]):
        return cls.from_operator_ndarray(np.array(operators, dtype=op.Operator))

    @classmethod
    def from_operator_ndarray(cls, operators: NDArray):

        return cls(cls._operator_ndarray_to_wpaulis(operators))

    @staticmethod
    def _operator_ndarray_to_wpaulis(operators) -> wpa.WeightedPauliArray:

        num_qubits = operators.flat[0].num_qubits

        all_num_terms = np.zeros(operators.shape, dtype=np.int_)
        for idx in np.ndindex(operators.shape):
            operator: op.Operator = operators[idx]
            if operator.num_qubits != num_qubits:
                raise ValueError("All the Operators must have the same number of qubits.")
            all_num_terms[idx] = operator.num_terms

        weights_shape = operators.shape + (all_num_terms.max(),)
        strings_shape = weights_shape + (num_qubits,)

        new_weights = np.zeros(weights_shape, dtype=np.complex_)
        new_xstrings = np.zeros(strings_shape, dtype=np.bool_)
        new_zstrings = np.zeros(strings_shape, dtype=np.bool_)

        for idx in np.ndindex(operators.shape):
            operator: op.Operator = operators[idx]
            new_weights[*idx, : all_num_terms[idx]] = operator.weights
            new_xstrings[*idx, : all_num_terms[idx], :] = operator.paulis.x_strings
            new_zstrings[*idx, : all_num_terms[idx], :] = operator.paulis.z_strings

        return wpa.WeightedPauliArray.from_z_strings_and_x_strings_and_weights(new_zstrings, new_xstrings, new_weights)


def commutator(
    operators_1: OperatorArrayType1,
    operators_2: OperatorArrayType1,
    combine_repeated_terms=False,
    remove_small_weights=False,
) -> OperatorArrayType1:
    assert is_broadcastable(operators_1.shape, operators_2.shape)

    new_shape = broadcast_shape(operators_1.shape, operators_2.shape)

    commutator_array = np.empty(new_shape, dtype=op.Operator)
    for idx in np.ndindex(new_shape):
        idx1 = broadcasted_index(operators_1.shape, idx)
        idx2 = broadcasted_index(operators_2.shape, idx)
        # idx1 = tuple([i if operators_1.shape[dim] > 1 else 0 for dim, i in enumerate(idx)])
        # idx2 = tuple([i if operators_2.shape[dim] > 1 else 0 for dim, i in enumerate(idx)])

        one_commutator = op.commutator(operators_1.get_operator(*idx1), operators_2.get_operator(*idx2))
        if combine_repeated_terms:
            one_commutator = one_commutator.combine_repeated_terms()
        if remove_small_weights:
            one_commutator = one_commutator.remove_small_weights()

        commutator_array[idx] = one_commutator

    return OperatorArrayType1.from_operator_ndarray(commutator_array)


def anticommutator(
    operators_1: OperatorArrayType1,
    operators_2: OperatorArrayType1,
    combine_repeated_terms=False,
    remove_small_weights=False,
) -> OperatorArrayType1:
    assert is_broadcastable(operators_1.shape, operators_2.shape)

    new_shape = broadcast_shape(operators_1.shape, operators_2.shape)

    commutator_array = np.empty(new_shape, dtype=op.Operator)
    for idx in np.ndindex(new_shape):
        idx1 = tuple([i if operators_1.shape[dim] > 1 else 0 for dim, i in enumerate(idx)])
        idx2 = tuple([i if operators_2.shape[dim] > 1 else 0 for dim, i in enumerate(idx)])

        operators_1.get_operator(*idx1)
        operators_2.get_operator(*idx2)

        one_commutator = op.anticommutator(operators_1.get_operator(*idx1), operators_2.get_operator(*idx2))
        if combine_repeated_terms:
            one_commutator = one_commutator.combine_repeated_terms()
        if remove_small_weights:
            one_commutator = one_commutator.remove_small_weights()

        commutator_array[idx] = one_commutator

    return OperatorArrayType1.from_operator_ndarray(commutator_array)


def concatenate(operatorss: Tuple[OperatorArrayType1, ...], axis: int) -> OperatorArrayType1:
    assert is_concatenatable(operatorss, axis)

    wpauliss = tuple(operators.wpaulis for operators in operatorss)

    return OperatorArrayType1(wpa.concatenate(wpauliss, axis=axis))
