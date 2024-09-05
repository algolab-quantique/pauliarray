from typing import TYPE_CHECKING, Any, Callable, List, Self, Tuple, Union

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
    def weights(self) -> "np.ndarray[np.complex]":
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

    def __eq__(self, other: Self) -> "np.ndarray[np.bool]":
        """
        Checks element-wise if the operators in the array are equal to the other.

        Args:
            other (OperatorArrayType1): Another OperatorArrayType1. Must be broadcastable.

        Returns:
            "np.ndarray[np.bool]": An array indicating where the operators are equal.
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
        Adds another operator array to the current operator array.
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
        Adds another OperatorArrayType1 to the current operator array.

        Args:
            other (OperatorArrayType1): The other OperatorArrayType1 to add.

        Returns:
            OperatorArrayType1: The resulting OperatorArrayType1.
        """
        new_wpaulis = wpa.concatenate([self.wpaulis, other.wpaulis], self.ndim)

        return OperatorArrayType1(new_wpaulis)

    def add_scalar(self, other: NDArray) -> Self:
        """
        Adds scalar array to the operator array.

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

    def x(self, qubits: Union[int, List[int]], inplace: bool = True) -> Self:
        """
        Applies X transformations on qubits of Operators. This leaves the Pauli Strings unchanged but produce
        phase factors -1 when operators are Y or Z.

        Args:
            qubits (int or list[int]): The qubits on which to apply the X.
            inplace (bool, optional): Applies the changes to self if True. Returns a modified copy if False.
            Defaults to True.

        Returns:
            OperatorArrayType1: A modified self if inplace=True, else returns a new modified instance of
            OperatorArrayType1.
        """

        if not inplace:
            return self.copy().x(qubits)

        self.wpaulis.x(qubits, inplace=True)
        return self

    def h(self, qubits: Union[int, List[int]], inplace: bool = True) -> Self:
        """
        Applies a H transformation on qubits of OperatorArray. This exchanges X matrices for Z matrices and vice-versa.
        It exchanges Y matrices into -Y matrices.

        Args:
            qubits (Union[int, List[int]]): The qubits to apply the H transformation to.
            inplace (bool, optional): Applies the changes to self if True. Returns a modified copy if False.
            Defaults to True.

        Returns:
            OperatorArrayType1: A modified self if inplace=True, else returns a new modified instance of
            OperatorArrayType1.
        """
        if not inplace:
            return self.copy().h(qubits)

        self.wpaulis.h(qubits, inplace=True)
        return self

    def s(self, qubits: Union[int, List[int]], inplace: bool = True) -> Self:
        """
        Applies S transformations on qubits of Operator. This exchanges X for Y and vice-versa with respective factors.

        Args:
            qubits (int or list[int]): The qubits on which to apply the S.
            inplace (bool, optional): Applies the changes to self if True. Returns a modified copy if False.
            Defaults to True.

        Returns:
            OperatorArrayType1: A modified self if inplace=True, else returns a new modified instance of
            OperatorArrayType1.
        """
        if not inplace:
            return self.copy().s(qubits)

        self.wpaulis.s(qubits, inplace=True)
        return self

    def cx(
        self, control_qubits: Union[int, List[int]], target_qubits: Union[int, List[int]], inplace: bool = True
    ) -> Self:
        """
        Applies CX transformations on qubits of Operator. The order of the CX is set by the order of the qubits.

        Args:
            control_qubits (int or list[int]): The qubits which controls the CZ.
            target_qubits (int or list[int]): The qubits target by CZ.
            inplace (bool, optional): Applies the changes to self if True. Returns a modified copy if False.
            Defaults to True.

        Returns:
            OperatorArrayType1: A modified self if inplace=True, else returns a new modified instance of
            OperatorArrayType1.
        """
        if not inplace:
            return self.copy().cx(control_qubits, target_qubits)

        self.wpaulis.cx(control_qubits, target_qubits, inplace=True)
        return self

    def cz(
        self, control_qubits: Union[int, List[int]], target_qubits: Union[int, List[int]], inplace: bool = True
    ) -> Self:
        """
        Applies CZ transformations on qubits of Operator. The order of the CZ is set by the order of the qubits.

        Args:
            control_qubits (int or list[int]): The qubits which controls the CZ.
            target_qubits (int or list[int]): The qubits target by CZ.
            inplace (bool, optional): Applies the changes to self if True. Returns a modified copy if False.
            Defaults to True.

        Returns:
            OperatorArrayType1: A modified self if inplace=True, else returns a new modified instance of
            OperatorArrayType1.
        """
        if not inplace:
            return self.copy().cz(control_qubits, target_qubits)

        self.wpaulis.cz(control_qubits, target_qubits, inplace=True)
        return self

    def clifford_conjugate(self, clifford: "Operator", inplace: bool = True) -> Self:
        """
        Performs a Clifford transformation.

        Args:
            clifford (Operator) : Must represent a Clifford transformation with the correct number of qubits.
            inplace (bool, optional): Applies the changes to self if True. Returns a modified copy if False.
            Defaults to True.

        Returns:
            OperatorArrayType1: The transformed OperatorArrayType1.
        """

        new_wpaulis = self.wpaulis.clifford_conjugate(clifford)

        return OperatorArrayType1(new_wpaulis)

    def expectation_values_from_paulis(
        self, paulis_expectation_values: NDArray[np.float64]
    ) -> "np.ndarray[np.complex]":
        """
        Returns the Operator array expectation value given the expectation values of the Paulis.

        Args:
            paulis_expectation_values (NDArray[float]): An array of expectation values for each Paulis present in the
            OperatorArray.

        Returns:
            NDArray: An array of complex expectation values for the Operator array, derived from the given
            Paulis expectation values.
        """

        return np.sum(self.wpaulis.expectation_values_from_paulis(paulis_expectation_values), axis=-1)

    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float64]) -> "np.ndarray[np.complex]":
        """
        Returns the Operator array covariances given the covariances of the Paulis.

        Args:
            paulis_covariances (NDArray[float]): An array of covariances for each Paulis present in the
            OperatorArray.

        Returns:
            NDArray: An array of covariances for the Operator array, derived from the given Paulis covariances.
        """

        return np.sum(self.wpaulis.covariances_from_paulis(paulis_covariances), axis=(self.ndim, -1))

    def combine_repeated_terms(self, inplace=False) -> Self:
        """
        Combines repeated terms within each operator in the array.

        Args:
            inplace (bool, optional): Applies the changes to self if True. Returns a modified copy if False.
            Defaults to False

        Returns:
            OperatorArrayType1 : A modified self if inplace=True, else returns a new modified instance of
            OperatorArrayType1.
        """

        red_operators = np.empty(self.shape, dtype=op.Operator)
        for idx in np.ndindex(self.shape):
            red_operators[idx] = self.get_operator(*idx).combine_repeated_terms()

        if inplace:
            new_wpaulis = self._operator_ndarray_to_wpaulis(red_operators)
            self._wpaulis = new_wpaulis
            return self

        return OperatorArrayType1.from_operator_ndarray(red_operators)

    def filter_weights(self, filter_function: Callable) -> Self:
        """
        Filters the weights of the Pauli terms using a provided filter function.

        Args:
            filter_function (Callable): A function that takes an array of weights and returns a boolean mask array of
            the same shape, indicating which weights to keep.

        Returns:
            OperatorArrayType1: A new instance of the class with the filtered weights applied to the Pauli terms.
        """
        weight_filter_mask = filter_function(self.weights)

        all_num_terms = np.sum(weight_filter_mask, axis=-1)
        max_num_terms = all_num_terms.max()

        new_wpaulis = wpa.WeightedPauliArray.new(self.shape + (max_num_terms,), self.num_qubits)
        for idx in np.ndindex(self.shape):
            new_wpaulis[idx, : all_num_terms[idx]] = self.wpaulis[idx, weight_filter_mask[idx]]

        return OperatorArrayType1(new_wpaulis)

    def remove_small_weights(self, threshold: float = 1e-14) -> Self:
        """
        Removes Pauli terms with weights smaller than a specified threshold.

        Args:
            threshold (float, optional): The threshold below which Pauli term weights will be removed.
            Defaults to 1e-14.

        Returns:
            OperatorArrayType1: A new instance of the class with the small weights removed.
        """
        return self.filter_weights(lambda weight: np.abs(weight) > threshold)

    def sum(self, axes: Union[Tuple[int, ...], None] = None) -> Union[op.Operator, Self]:
        """
        Performs summation along the specified axes, combining these axes with the hidden summation axis.
        Returns an Operator if a single operator remains in the operator array. Else, return an OperatorArray.

        Args:
            axes (Union[Tuple[int, ...], None], optional): Axes along which to sum. If None, the entire array is summed.
                Axes can be negative to count from the last to the first axis. Defaults to None.

        Returns:
            Union[op.Operator, OperatorArrayType1]: An instance of `op.Operator` if the result is a single operator,
            otherwise an instance of OperatorArrayType1 with the summed array.
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

    @classmethod
    def from_pauli_array(cls, paulis: pa.PauliArray, summation_axis: Union[Tuple[int, ...], None] = None) -> Self:
        """
        Converts a PauliArray into an OperatorArrayType1.

        Args:
            paulis (pa.PauliArray): The input PauliArray to convert.
            summation_axis (int, optional): Which axis of PauliArray to use for summation.
            If None, each Pauli string becomes an operator. Defaults to None.

        Returns:
            OperatorArrayType1: A new instance of OperatorArrayType1 according to given Paulis.
        """
        new_paulis = paulis.reshape(paulis.shape + (1,))

        new_operators = cls(wpa.WeightedPauliArray.from_paulis(new_paulis))

        if summation_axis is not None:
            new_operators = new_operators.sum(summation_axis)

        return new_operators

    @classmethod
    def from_weighted_pauli_array(
        cls, wpaulis: wpa.WeightedPauliArray, summation_axis: Union[Tuple[int, ...], None] = None
    ) -> Self:
        """
        Converts a WeightedPauliArray into an OperatorArrayType1.

        Args:
            wpaulis (wpa.WeightedPauliArray): The input WeightedPauliArray to convert.
            summation_axis (int, optional): Which axis of WeightedPauliArray to use for summation.
            If None, each Pauli string becomes an operator. Defaults to None.

        Returns:
            OperatorArrayType1: A new instance of OperatorArrayType1 according to given Paulis.
        """
        new_wpaulis = wpaulis.reshape(wpaulis.shape + (1,))

        new_operators = cls(new_wpaulis)

        if summation_axis is not None:
            new_operators = new_operators.sum(summation_axis)

        return new_operators

    @classmethod
    def from_operator_list(cls, operators: List[op.Operator]) -> Self:
        """
        Converts a list of Operator into an OperatorArrayType1.

        Args:
            operators (List[op.Operator]): The input list of operators to convert.

        Returns:
            OperatorArrayType1: A new instance of OperatorArrayType1 according to given operators.
        """
        return cls.from_operator_ndarray(np.array(operators, dtype=op.Operator))

    @classmethod
    def from_operator_ndarray(cls, operators: NDArray):
        """
        Converts an NDArray of operators into an OperatorArrayType1.

        Args:
            operators (NDArray): The input array of operators to convert.

        Returns:
            OperatorArrayType1: A new instance of OperatorArrayType1 according to given operators.
        """
        return cls(cls._operator_ndarray_to_wpaulis(operators))

    @staticmethod
    def _operator_ndarray_to_wpaulis(operators) -> wpa.WeightedPauliArray:
        """
        Converts an NDArray of operators into a WeightedPauliArray.

        Args:
            operators (NDArray): The input array of operators to convert.

        Returns:
            wpa.WeightedPauliArray: A new instance of WeightedPauliArray according to given operators.
        """
        num_qubits = operators.flat[0].num_qubits

        all_num_terms = np.zeros(operators.shape, dtype=np.int_)
        for idx in np.ndindex(operators.shape):
            operator: op.Operator = operators[idx]
            if operator.num_qubits != num_qubits:
                raise ValueError("All the Operators must have the same number of qubits.")
            all_num_terms[idx] = operator.num_terms

        weights_shape = operators.shape + (all_num_terms.max(),)
        strings_shape = weights_shape + (num_qubits,)

        new_weights = np.zeros(weights_shape, dtype=np.complex128)
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
    r"""
    Computes the commutator

    .. math::
        [A, B] = AB - BA

    for pairs of operators from two operator arrays.

    Args:
        operators_1 (OperatorArrayType1): The first array of operators.
        operators_2 (OperatorArrayType1): The second array of operators.
        combine_repeated_terms (bool, optional): If True, combines repeated terms in the resulting commutators.
        Defaults to False.
        remove_small_weights (bool, optional): If True, removes small weights from the resulting commutators.
        Defaults to False.

    Returns:
        OperatorArrayType1: An array of commutators of the input operator arrays.
    """
    assert is_broadcastable(operators_1.shape, operators_2.shape)

    new_shape = broadcast_shape(operators_1.shape, operators_2.shape)

    commutator_array = np.empty(new_shape, dtype=op.Operator)
    for idx in np.ndindex(new_shape):
        idx1 = broadcasted_index(operators_1.shape, idx)
        idx2 = broadcasted_index(operators_2.shape, idx)

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
    r"""
    Computes the anticommutator

    .. math::
        {A, B} = AB + BA

    for pairs of operators from two operator arrays.

    Args:
        operators_1 (OperatorArrayType1): The first array of operators.
        operators_2 (OperatorArrayType1): The second array of operators.
        combine_repeated_terms (bool, optional): If True, combines repeated terms in the resulting anticommutators. Defaults to False.
        remove_small_weights (bool, optional): If True, removes small weights from the resulting anticommutators. Defaults to False.

    Returns:
        OperatorArrayType1: An array of anticommutators of the input operator arrays.
    """
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


def concatenate(operators: Tuple[OperatorArrayType1, ...], axis: int) -> OperatorArrayType1:
    """
    Concatenates multiple operator arrays along the specified axis.

    Args:
        operators (Tuple[OperatorArrayType1, ...]): A tuple of operator arrays to concatenate.
        axis (int): The axis along which to concatenate the operator arrays.

    Returns:
        OperatorArrayType1: A new operator array resulting from the concatenation.
    """
    assert is_concatenatable(operators, axis)

    wpauliss = tuple(operators.wpaulis for operators in operators)

    return OperatorArrayType1(wpa.concatenate(wpauliss, axis=axis))
