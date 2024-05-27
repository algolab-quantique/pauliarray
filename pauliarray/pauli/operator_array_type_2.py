from numbers import Number
from typing import Any, Callable, List, Tuple, Union, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
from pauliarray.utils.array_operations import broadcast_shape, is_broadcastable, is_concatenatable


class OperatorArrayType2(object):
    def __init__(self, basis_paulis: pa.PauliArray, weights: NDArray[np.complex_]):
        """
        Initializes an OperatorArrayType2 object.

        Args:
            basis_paulis (pa.PauliArray): The basis Pauli arrays.
            weights (NDArray[np.complex_]): The weights associated with the Pauli arrays.
        """
        assert basis_paulis.size == weights.shape[-1]

        self._basis_paulis = basis_paulis
        self._weights = weights

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Gets the shape of the OperatorArrayType2 excluding the last dimension (summation dimension).

        Returns:
            tuple: The shape of the weights excluding the last dimension.
        """
        return self._weights.shape[:-1]

    @property
    def size(self) -> int:
        """
        Gets the total number of elements in the OperatorArrayType2 excluding the last dimension (summation dimension).

        Returns:
            int: The total number of elements.
        """
        return np.prod(self.shape)

    @property
    def basis_paulis(self) -> pa.PauliArray:
        """
        Gets the basis Pauli arrays.

        Returns:
            pa.PauliArray: The basis Pauli arrays.
        """
        return self._basis_paulis

    @property
    def weights(self) -> NDArray[np.complex_]:
        """
        Gets the weights associated with the basis Pauli arrays.

        Returns:
            NDArray[np.complex_]: The weights.
        """
        return self._weights

    @property
    def paulis(self) -> pa.PauliArray:
        """
        Gets the basis Pauli arrays.

        Returns:
            pa.PauliArray: The basis Pauli arrays.
        """
        return self.basis_paulis

    @property
    def num_qubits(self) -> int:
        """
        Gets the number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self._basis_paulis.num_qubits

    @property
    def ndim(self) -> int:
        """
        Gets the number of dimensions of the weights excluding the last dimension.

        Returns:
            int: The number of dimensions.
        """
        return len(self.shape)

    def __getitem__(self, key) -> Self:
        """
        Gets an item from the weights using the provided key.

        Args:
            key: The key to index the weights.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 object with the indexed weights.
        """
        return OperatorArrayType2(self.basis_paulis.copy(), self.weights[key])

    def _mul(self, other: Union[Number, ArrayLike]) -> Self:
        """
        Multiplies the weights by a number or an array.

        Args:
            other (Union[Number, ArrayLike]): The number or array to multiply the weights by.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 object with the multiplied weights.
        """
        if isinstance(other, (Number, ArrayLike)):
            return self.mul_weights(other)

        return NotImplemented

    def reshape(self, shape: Tuple[int, ...]) -> Self:
        """
        Returns a copy of self with a new shape.

        Args:
            shape (tuple[int]): Tuple containing the new shape.

        Returns:
            OperatorArrayType2: The OperatorArray object with the new shape.
        """
        return OperatorArrayType2(self.basis_paulis.copy(), self.weights.reshape(shape))

    def flatten(self) -> Self:
        """
        Returns a flattened copy of self.

        Returns:
            OperatorArrayType2: A flattened copy of the current OperatorArray.
        """
        shape = (np.prod(self.shape),)
        return self.reshape(shape)

    def squeeze(self) -> Self:
        """
        Returns an OperatorArray with axes of length one removed.

        Returns:
            OperatorArrayType2: The squeezed OperatorArray.
        """
        return OperatorArrayType2(self.basis_paulis.copy(), self.weights.squeeze())

    def __mul__(self, other: Any) -> Self:
        """
        Multiplies the weights by a number.

        Args:
            other (any): The number to multiply the weights by.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 object with the multiplied weights.
        """
        if isinstance(other, Number):
            return self.mul_weights(other)

        return NotImplemented

    def __rmul__(self, other: Any) -> Self:
        """
        Multiplies the weights by a number.

        Args:
            other (any): The number to multiply the weights by.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 object with the multiplied weights.
        """
        if isinstance(other, Number):
            return self.mul_weights(other)

        return NotImplemented

    def mul_weights(self, other: Union[Number, NDArray]) -> Self:
        """
        Multiplies the weights by a number or an array.

        Args:
            other (Union[Number, NDArray]): The number or array to multiply the weights by.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 object with the multiplied weights.
        """
        other = np.array(other)
        new_weights = self.weights * other[..., None]
        return OperatorArrayType2(self.basis_paulis.copy(), new_weights)

    def compose(self, other: Any) -> Any:
        """
        Composes the current operator with another operator.

        Args:
            other (Any): The other operator to compose with.

        Returns:
            Any: The composed operator.
        """
        if isinstance(other, OperatorArrayType2):
            return self.compose_operator_array_type_2(other)

        return NotImplemented

    def compose_operator_array_type_2(self, other: Self) -> Self:
        """
        Composes the current operator with another OperatorArrayType2.

        Args:
            other (OperatorArrayType2): The other OperatorArrayType2 to compose with.

        Returns:
            OperatorArrayType2: The composed OperatorArrayType2.
        """
        tmp_weights = self.weights[..., :, None] * other.weights[..., None, :]

        new_pauli_basis, factors = self.basis_paulis[:, None].mul_pauli_array(other.basis_paulis[None, :])

        new_shape = tmp_weights.shape[:-2] + (tmp_weights.shape[-1] * tmp_weights.shape[-2],)
        new_weights = (tmp_weights * factors).reshape(new_shape)

        new_operator_array = OperatorArrayType2(new_pauli_basis, new_weights).combine_basis_paulis()

        return new_operator_array

    def add(self, other: Any) -> Any:
        """
        Adds the current operator with another operator.

        Args:
            other (Any): The other operator to add.

        Returns:
            Any: The summed operator.
        """
        if isinstance(other, OperatorArrayType2):
            return self.add_operator_array_type_2(other)

        return NotImplemented

    def add_operator_array_type_2(self, other: Self) -> Self:
        """
        Adds the current operator with another OperatorArrayType2.

        Args:
            other (OperatorArrayType2): The other OperatorArrayType2 to add.

        Returns:
            OperatorArrayType2: The summed OperatorArrayType2.
        """
        assert self.shape == other.shape

        new_weights = np.concatenate((self.weights, other.weights), axis=self.ndim)
        new_pauli_basis = pa.concatenate((self.basis_paulis, other.basis_paulis), axis=0)

        new_operator_array = OperatorArrayType2(new_pauli_basis, new_weights).combine_basis_paulis()

        return new_operator_array

    def filter_weights(self, filter_function: Callable) -> Self:
        """
        Filters the weights using a filter function.

        Args:
            filter_function (Callable): The function to filter the weights. Must return a mask appliable on weights.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 object with filtered weights.
        """
        weight_filter_mask = filter_function(self.weights)

        new_weights = np.zeros(self.shape + (self.basis_paulis.size,), dtype=complex)
        new_weights[weight_filter_mask] = self.weights[weight_filter_mask]

        return OperatorArrayType2(self.basis_paulis.copy(), new_weights)

    def remove_small_weights(self, threshold: float = 1e-12) -> Self:
        """
        Removes weights that are smaller than a threshold.

        Args:
            threshold (float): The threshold below which weights are removed. Defaults to 1e-12.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 object with small weights removed.
        """
        return self.filter_weights(lambda weight: np.abs(weight) > threshold)

    def remove_unused_basis_paulis(self) -> Self:
        """
        Removes Paulis from the basis that are not used in any operator of the OperatorArray.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 instance with unused Pauli operators removed from the basis.
        """
        keep_basis = np.any(self.weights != 0, axis=tuple(range(self.ndim)))

        if np.all(keep_basis):
            return self

        self._basis_paulis = self.basis_paulis[keep_basis]

        new_weights = self.weights[..., keep_basis]
        self._weights = new_weights

        return self

    def combine_basis_paulis(self) -> Self:
        """
        Combines repeated Pauli operators in the basis and updates weights accordingly.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 instance with combined Pauli operators and updated weights.
        """
        new_basis_paulis, inverse = pa.unique(self.basis_paulis, return_inverse=True)

        if new_basis_paulis.size == self.basis_paulis.size:
            return self

        # need to put the pauli_basis dimension first to use np.add.at
        new_weights_bw = np.zeros((new_basis_paulis.size,) + self.shape, dtype=complex)
        np.add.at(new_weights_bw, inverse, np.moveaxis(self.weights, -1, 0))

        self._weights = np.moveaxis(new_weights_bw, 0, -1)
        self._basis_paulis = new_basis_paulis

        return self

    def expectation_values_from_paulis(self, paulis_expectation_values: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Calculates the expectation values of the operators given the expectation values of the Pauli operators.

        Args:
            paulis_expectation_values (NDArray[float]): Expectation values of the input Paulis.

        Returns:
            NDArray[float]: The expectation values of the operators in operator array.
        """
        assert np.all(paulis_expectation_values.shape == self.paulis.shape)

        expectation_values = np.dot(self.weights, paulis_expectation_values)

        return expectation_values

    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Calculates the covariances of the operators given the covariances of the Pauli operators.

        Args:
            paulis_covariances (NDArray[float]): Covariances of the input Paulis.

        Returns:
            NDArray[float]: The covariances of the operators.
        """
        assert np.all(paulis_covariances.shape == (self.paulis.shape + self.paulis.shape))

        covariances = np.einsum("...i,...j,ij", self.weights, self.weights.conj(), paulis_covariances)

        return covariances

    @staticmethod
    def build_basis_paulis(operators: NDArray) -> Tuple[pa.PauliArray, NDArray[np.complex_]]:
        """
        Builds the basis and the basis map.
        The basis is a PauliArray that contains each of the Pauli strings appearing in the operators.
        The basis map has the same shape as the OperatorArray and contains an array of indices that can be used to
        construct the operator with relevant Pauli strings from the basis.

        Args:
            operators (NDArray): Array of operator objects.

        Returns:
            Tuple[PauliArray, NDArray[np.complex_]]: A tuple containing :
                - A PauliArray : All the Pauli Strings present in the OperatorArray. Forms the basis.
                - An NDArray: An array definining the operators of the OperatorArray using the indices as reference to
            the basis.
        """
        for idx in np.ndindex(operators.shape):
            operators[idx].combine_repeated_terms(inplace=True)

        all_paulis = pa.concatenate([operators[idx].paulis for idx in np.ndindex(operators.shape)], 0)
        # a list to know to which operator the pauli in all_paulis belongs
        operator_indices = np.concatenate(
            [
                np.ones(operators[idx].num_terms, dtype=int) * operator_index
                for operator_index, idx in enumerate(np.ndindex(operators.shape))
            ],
            0,
        )
        basis_paulis, inverse_basis = pa.unique(all_paulis, return_inverse=True)

        weights = np.zeros(operators.shape + (basis_paulis.size,), dtype=np.complex_)
        for operator_index, idx in enumerate(np.ndindex(operators.shape)):
            weights[idx][inverse_basis[operator_indices == operator_index]] = operators[idx].weights

        return basis_paulis, weights

    def get_operator(self, *idx) -> op.Operator:
        """
        Retrieves the operator at the specified index.

        Args:
            *idx: Indices specifying the position of the desired operator.

        Returns:
            op.Operator: The operator at the specified index.
        """
        raw_weights = self.weights[idx]
        pauli_idx = np.nonzero(raw_weights)[0]

        weights = raw_weights[pauli_idx]
        paulis = self.basis_paulis[pauli_idx]

        return op.Operator.from_paulis_and_weights(paulis, weights)

    def sum(self, axis: Union[Tuple[int, ...], None] = None) -> op.Operator:
        """
        Sums the operators along the specified axis.

        Args:
            axis (Union[Tuple[int, ...], None]): Axis or axes along which to sum the operators. If None, sums over all axes.

        Returns:
            op.Operator: The summed operator.
        """
        if axis is None:
            axis = tuple(range(self.ndim))

        weights = np.sum(self.weights, axis=axis)

        return op.Operator.from_paulis_and_weights(self.basis_paulis, weights)

    @classmethod
    def from_operator_ndarray(cls, operators: NDArray) -> Self:
        """
        Constructs an OperatorArrayType2 instance from an array of operators.

        Args:
            operators (NDArray): Array of operator objects.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 instance.
        """
        basis_paulis, weights = cls.build_basis_paulis(operators)

        return cls(basis_paulis, weights)

    @classmethod
    def from_operator_list(cls, operators: List[op.Operator]) -> Self:
        """
        Constructs an OperatorArrayType2 instance from a list of operators.

        Args:
            operators (List[op.Operator]): List of operator objects.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 instance.
        """
        return cls.from_operator_ndarray(np.array(operators, dtype=op.Operator))

    @classmethod
    def from_operator(cls, operator: op.Operator) -> Self:
        """
        Constructs an OperatorArrayType2 instance from a single operator.

        Args:
            operator (op.Operator): An operator object.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 instance.
        """
        return cls.from_operator_ndarray(np.array([operator], dtype=op.Operator))

    @classmethod
    def from_pauli_array(cls, paulis: pa.PauliArray) -> Self:
        """
        Constructs an OperatorArrayType2 instance from a Pauli array.

        Args:
            paulis (pa.PauliArray): A Pauli array object.

        Returns:
            OperatorArrayType2: A new OperatorArrayType2 instance.
        """
        weights = np.eye(paulis.size, dtype=complex)
        weights.reshape(paulis.shape + (paulis.size,))
        return cls(weights, paulis.copy())


def commutator(operators_1: Self, operators_2: Self) -> Self:
    r"""
    Computes the commutator

    .. math::
        [A, B] = AB - BA

    for pairs of operators from two operator arrays.

    Args:
        operators_1 (OperatorArrayType2): The first array of operators.
        operators_2 (OperatorArrayType2): The second array of operators.
    Returns:
        OperatorArrayType2: An array of commutators of the input operator arrays.
    """
    assert is_broadcastable(operators_1.shape, operators_2.shape)

    new_shape = broadcast_shape(operators_1.shape, operators_2.shape)

    raw_commutators, raw_factors = pa.commutator(operators_1.basis_paulis[:, None], operators_2.basis_paulis[None, :])
    commutators = raw_commutators.flatten()

    weights = (
        operators_1.weights[..., :, None]
        * operators_2.weights[..., None, :]
        * np.expand_dims(raw_factors, axis=tuple(range(operators_1.ndim)))
    ).reshape(new_shape + (commutators.size,))

    return OperatorArrayType2(commutators, weights).combine_basis_paulis()


def concatenate(operatorss: Tuple[Self, ...], axis: int) -> Self:
    """
    Concatenates multiple operator arrays along the specified axis.

    Args:
        operators (Tuple[OperatorArrayType2, ...]): A tuple of operator arrays to concatenate.
        axis (int): The axis along which to concatenate the operator arrays.

    Returns:
        OperatorArrayType2: A new operator array resulting from the concatenation.
    """
    assert is_concatenatable(operatorss, axis)

    new_pauli_basis = pa.concatenate(tuple(operators.basis_paulis for operators in operatorss), axis=axis)

    basis_size = new_pauli_basis.size

    cum_basis_length = 0
    basis_paded_weightss = []
    for operators in operatorss:
        current_basis_size = operators.basis_paulis.size
        a_zeros = np.zeros(operators.shape + (cum_basis_length,))
        cum_basis_length += current_basis_size
        b_zeros = np.zeros(operators.shape + (basis_size - cum_basis_length,))

        basis_paded_weights = np.concatenate((a_zeros, operators.weights, b_zeros), axis=-1)
        basis_paded_weightss.append(basis_paded_weights)

    new_weights = np.concatenate(basis_paded_weightss, axis=axis)

    return OperatorArrayType2(new_pauli_basis, new_weights)
