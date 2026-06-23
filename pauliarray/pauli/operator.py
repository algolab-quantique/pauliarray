from numbers import Number
from typing import Any, Callable, List, Literal, Self, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.binary import bit_operations as bitops
from pauliarray.binary import symplectic
from pauliarray.utils.pauli_array_library import gen_complete_pauli_array_basis


class Operator(object):
    """
    Defines an operator as a linear combination of Pauli strings.
    """

    def __init__(self, weighted_paulis: wpa.WeightedPauliArray):
        """
        Initializes the Operator with a WeightedPauliArray.

        Args:
            weighted_paulis (WeightedPauliArray): A WeightedPauliArray object.
        """
        self._wpaulis = weighted_paulis.flatten()

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits.

        Returns:
            int: Number of qubits.
        """
        return self._wpaulis.num_qubits

    @property
    def num_terms(self) -> int:
        """
        Returns the number of terms.

        Returns:
            int: Number of terms.
        """
        return self._wpaulis.size

    @property
    def wpaulis(self) -> wpa.WeightedPauliArray:
        """
        Returns the WeightedPauliArray.

        Returns:
            WeightedPauliArray: The WeightedPauliArray object.
        """
        return self._wpaulis

    @property
    def weights(self) -> "np.ndarray[np.complex]":
        """
        Returns the weights of the Pauli terms.

        Returns:
            "np.ndarray[np.complex]": The weights.
        """
        return self._wpaulis.weights

    @property
    def paulis(self) -> pa.PauliArray:
        """
        Returns the PauliArray.

        Returns:
            PauliArray: The PauliArray object.
        """
        return self._wpaulis.paulis

    def __str__(self):
        """
        Returns a string representation of the Operator.

        Returns:
            str: String representation of the Operator.
        """
        return f"Operator: num_qubits = {self.num_qubits}, num_terms = {self.num_terms}, ..."

    def __add__(self, other: Union[Self, Number]) -> Self:
        """
        Adds another Operator or a scalar to this Operator.

        Args:
            other (Union[Operator, Number]): Another Operator or a scalar.

        Returns:
            Operator: The resulting Operator.
        """
        if isinstance(other, Operator):
            return self.add_operator(other)

        if isinstance(other, Number):
            return self.add_scalar(other)

        return NotImplemented

    def __mul__(self, other: any):
        """
        Multiplies this Operator with another Operator or a scalar.

        Args:
            other (Any): Another Operator or a scalar.

        Returns:
            Any: The resulting Operator or NotImplemented.
        """
        if isinstance(other, Operator):
            return self.compose_operator(other)

        if isinstance(other, Number):
            return self.mul_scalar(other)

        return NotImplemented

    def __rmul__(self, other: any):
        """
        Multiplies a scalar with this Operator from the right.

        Args:
            other (Any): A scalar.

        Returns:
            Any: The resulting Operator or NotImplemented.
        """
        if isinstance(other, Number):
            return self.mul_scalar(other)

        return NotImplemented

    def __eq__(self, other: Self) -> bool:
        """
        Checks if equal to another Operator. Two Operators are equal if after simplification their underlying WeightedPauliArrays are equal.

        Args:
            other (Operator): Another Operator.

        Returns:
            bool: True if the Operators are equal.
        """
        simple_self = self.combine_repeated_terms().remove_small_weights()
        simple_other = other.combine_repeated_terms().remove_small_weights()

        if simple_self.num_terms != simple_other.num_terms:
            return False

        self_order = pa.argsort(simple_self.paulis)
        other_order = pa.argsort(simple_other.paulis)

        return np.all(simple_self.wpaulis[self_order] == simple_other.wpaulis[other_order])

    def copy(self) -> Self:
        """
        Returns a copy of the Operator.

        Returns:
            Operator: A copy of the Operator.
        """
        return Operator(self._wpaulis.copy())

    def adjoint(self) -> Self:
        """
        Returns the adjoint of the Operator.

        Returns:
            Operator: The adjoint Operator.
        """
        new_wpaulis = wpa.WeightedPauliArray(self.wpaulis.paulis.copy(), np.conj(self.wpaulis.weights))
        return Operator(new_wpaulis)

    def replace_paulis(self, new_paulis: pa.PauliArray, inplace: bool = False) -> Self:
        """
        Replace the Paulis in the object by the new ones.

        Args:
            new_paulis (PauliArray): The new Paulis
            inplace (bool, optional): If True replace inside the current instance. If False returns a new instance. Defaults to False.

        Returns:
            Operator: The object with replaced Paulis
        """

        assert new_paulis.shape == self.wpaulis.shape

        if inplace:
            self._wpaulis._paulis = new_paulis.copy()
            return self

        return Operator.from_paulis_and_weights(new_paulis.copy(), self.weights.copy())

    def replace_weights(self, new_weights: NDArray, inplace: bool = False) -> Self:
        """
        Replace the Weights in the object by the new ones.

        Args:
            new_weights (NDArray): The new Weights
            inplace (bool, optional): If True replace inside the current instance. If False returns a new instance. Defaults to False.

        Returns:
            Operator: The object with replaced weights
        """

        assert new_weights.shape == self.wpaulis.shape

        if inplace:
            self._wpaulis._weights = new_weights.copy()
            return self

        return Operator.from_paulis_and_weights(self.paulis.copy(), new_weights.copy())

    def replace_paulis_and_weights(
        self, new_paulis: pa.PauliArray, new_weights: NDArray, inplace: bool = False
    ) -> Self:
        """
        Replace the Paulis and Weights in the object by the new ones.

        Args:
            new_paulis (PauliArray): The new Paulis
            new_weights (NDArray): The new Weights
            inplace (bool, optional): If True replace inside the current instance. If False returns a new instance. Defaults to False.

        Returns:
            Operator: The object with replaced Paulis and weights
        """

        assert new_paulis.shape == self.wpaulis.shape
        assert new_weights.shape == self.wpaulis.shape

        if inplace:
            self._wpaulis._paulis = new_paulis.copy()
            self._wpaulis._weights = new_weights.copy()
            return self

        return Operator.from_paulis_and_weights(new_paulis.copy(), new_weights.copy())

    def clifford_transform_paulis(self, clifford_fct: Callable, *args, inplace=False) -> Self:
        """
        Transform the Paulis using a Clifford transformation (from transformation.cliffords)

        Args:
            clifford_fct (Callable): A Clifford function (from transformation.cliffords)
            *args: The arguments of the Clifford function, such as the qubits on which to apply.
            inplace (bool, optional): If True replace the existing Paulis. Defaults to False.

        Returns:
            Operator: The transformed Operator
            "np.ndarray[np.complex]": The factors resulting from the transformation
        """

        new_paulis, factors = clifford_fct(self.paulis, *args)
        new_weights = self.weights * factors

        return self.replace_paulis_and_weights(new_paulis, new_weights, inplace)

    def take_qubits(self, indices: Union["np.ndarray[np.int]", range, int]) -> Self:
        """
        Takes a subset of qubits.

        Args:
            indices (Union["np.ndarray[np.int]", range, int]): Indices of the qubits to take.

        Returns:
            Operator: The resulting Operator with the subset of qubits.
        """
        if isinstance(indices, int):
            indices = np.array([indices], dtype=int)

        new_wpaulis = self.wpaulis.take_qubits(indices)

        return Operator(new_wpaulis)

    def compress_qubits(self, condition: "np.ndarray[np.bool]") -> Self:
        """
        Compresses the qubits based on the given condition.

        Args:
            condition ("np.ndarray[np.bool]"): Condition for compressing the qubits.

        Returns:
            Operator: The resulting Operator with compressed qubits.
        """
        new_wpaulis = self.wpaulis.compress_qubits(condition)
        return Operator(new_wpaulis)

    def compose(self, other: Any) -> Any:
        """
        Composes the Operator with another Operator.

        Args:
            other (Any): Another Operator.

        Returns:
            Any: The resulting Operator or NotImplemented.
        """
        if isinstance(other, Operator):
            return self.compose_operator(other)

        return NotImplemented

    def compose_operator(self, other: Self) -> Self:
        """
        Composes the Operator with another Operator.

        Args:
            other (Operator): Another Operator.

        Returns:
            Operator: The resulting Operator.
        """
        new_wpaulis = self.wpaulis[:, None].compose_weighted_pauli_array(other.wpaulis[None, :])
        return Operator(new_wpaulis).combine_repeated_terms()

    def mul_scalar(self, other: Number):
        """
        Multiplies the Operator by a scalar.

        Args:
            other (Number): A scalar.

        Returns:
            Operator: The resulting Operator.
        """
        new_wpaulis = self.wpaulis.mul_weights(other)
        return Operator(new_wpaulis)

    def power(self, exponent: int, simplify: bool = False) -> Self:
        """
        Raises the Operator to the specified exponent.

        Args:
            exponent (int): The exponent to raise the Pauli operator to.
            simplify (bool): If True, the Operator is simplified at every multiplication with itself. This can improve performance. Defaults to False.

        Returns:
            Operator: A new Pauli operator resulting from raising the original operator to the exponent.

        Raises:
            ValueError: If the exponent is negative.
        """
        if exponent < 0:
            raise ValueError("The power must be a positive integer.")

        if exponent == 0:
            return Operator.identity(self.num_qubits)

        result = self

        for _ in range(exponent - 1):
            result = result.compose_operator(self)

            if simplify:
                result = result.simplify()

        return result

    def tensor(self, other: Any) -> Any:
        """
        Takes the tensor product of the Operator with another Operator.

        Args:
            other (Any): Another Operator.

        Returns:
            Any: The resulting Operator or NotImplemented.
        """
        if isinstance(other, Operator):
            return self.tensor_operator(other)

        return NotImplemented

    def tensor_operator(self, other: Self) -> Self:
        """
        Takes the tensor product of the Operator with another Operator.

        Args:
            other (Operator): Another Operator.

        Returns:
            Operator: The resulting Operator.
        """
        new_shape = (self.num_terms, other.num_terms)

        exp_self = wpa.broadcast_to(wpa.expand_dims(self.wpaulis, 1), new_shape)
        exp_other = wpa.broadcast_to(wpa.expand_dims(other.wpaulis, 0), new_shape)

        new_wpaulis = exp_self.tensor_weighted_pauli_array(exp_other)

        return Operator(new_wpaulis)

    def add_operator(self, other: Self) -> Self:
        """
        Adds another Operator to this Operator.

        Args:
            other (Operator): Another Operator.

        Returns:
            Operator: The resulting Operator.
        """
        new_wpaulis = wpa.concatenate([self.wpaulis, other.wpaulis], 0)
        return Operator(new_wpaulis).combine_repeated_terms()

    def add_scalar(self, scalar: Number) -> Self:
        """
        Adds a scalar to this Operator.

        Args:
            scalar (Number): A scalar.

        Returns:
            Operator: The resulting Operator.
        """
        return self.add_operator(Operator.identity(self.num_qubits).mul_scalar(scalar))

    def inspect(self) -> str:
        """
        Generates a string representation of the Operator showing sum of weighted Pauli strings.

        Returns:
            str: A string representation of the Operator.
        """
        labels = self.wpaulis.paulis.to_labels()
        weights = self.wpaulis.weights

        detail_str = "Operator\nSum of\n"
        detail_str += wpa.WeightedPauliArray.label_table_2d(labels[:, None], weights[:, None])

        return detail_str

    def clifford_conjugate(self, clifford: Self, inplace: bool = True) -> Self:
        """
        Performs a Clifford conjugate transformation on the Operator.

        Args:
            clifford (Operator): The Clifford operator.
            inplace (bool): If True, applies changes to self; otherwise, returns a modified copy.

        Returns:
            Operator: The transformed Operator.
        """
        new_paulis, factors = clifford.clifford_conjugate_pauli_array_old(self.paulis)
        new_weights = self.weights * factors
        if inplace:
            self._wpaulis._paulis = new_paulis
            self._wpaulis._weights = new_weights

            return self

        return Operator.from_paulis_and_weights(new_paulis, new_weights)

    def clifford_conjugate_pauli_array_old(
        self, other: pa.PauliArray
    ) -> Tuple[pa.PauliArray, "np.ndarray[np.complex]"]:
        """
        Transform a PauliArray using self to perform a Clifford conjugate.

        Args:
            other (pa.PauliArray): A PauliArray

        Returns:
            pa.PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": Residual coefficient
        """

        assert self.is_clifford()

        self_phases = np.mod(np.round(-np.angle(self.weights) / (np.pi / 2)).astype(int), 4)

        original_shape = other.shape
        flat_orig_paulis = other.flatten()

        # dim 0 = pauli
        # dim 1 = left side
        # dim 2 = right side

        prod_paulis_1, phases1 = self.wpaulis.paulis[None, :].compose_pauli_array(flat_orig_paulis[:, None])
        prod_paulis, phases2 = prod_paulis_1[:, :, None].compose_pauli_array(self.wpaulis.paulis[None, None, :])

        phases = np.mod(phases1[:, :, None] + phases2 - self_phases[None, :, None] + self_phases[None, None, :], 4)

        new_paulis = pa.PauliArray.identities(flat_orig_paulis.shape, flat_orig_paulis.num_qubits)
        new_phases = np.zeros(flat_orig_paulis.shape, dtype=phases.dtype)

        for i in range(phases.shape[0]):

            flat_prod_paulis = prod_paulis[i, :, :].flatten()
            flat_phases = phases[i, :, :].flatten()

            u_prods_phases = np.zeros_like(phases[i, :, :])

            u_prod_paulis, inverse = pa.fast_flat_unique(flat_prod_paulis, return_inverse=True)

            for j in range(phases.shape[1]):
                u_prods_phases[j, :] = flat_phases[inverse == j]

            prod_mask_result = np.all(u_prods_phases[:, [0]] == u_prods_phases, axis=1)

            assert np.sum(prod_mask_result) == 1, "The Operator is probably not Clifford"

            new_paulis[i] = u_prod_paulis[prod_mask_result]
            new_phases[i] = u_prods_phases[prod_mask_result, 0]

        return (
            new_paulis.reshape(original_shape),
            new_phases.reshape(original_shape),
        )

    def clifford_conjugate_pauli_array(self, other: pa.PauliArray) -> Tuple[pa.PauliArray, NDArray]:
        """
        Transform a PauliArray using self to perform a Clifford conjugate.
        (Faster prototype)

        Args:
            other (pa.PauliArray): A PauliArray

        Returns:
            pa.PauliArray: The transformed PauliArray
            "np.ndarray[np.complex]": Residual coefficient
        """

        assert self.is_clifford()

        self_amplitude = 1 / self.num_terms
        self_phases = np.mod(np.round(-np.angle(self.weights) / (np.pi / 2)).astype(int), 4)

        # print("self_phase")
        # print(self_phases)

        # flatten, will be reshape at the end
        original_shape = other.shape
        flat_other = other.flatten()

        # check the commutation between the paulis in other and in self
        commute_mask = flat_other[:, None].commute_with(self.paulis[None, :])

        idem_mask = np.all(commute_mask == commute_mask[:, 0], axis=1)
        change_mask = ~idem_mask

        print(idem_mask.astype(int))

        # computes all the product between the paulis in self into a square array
        # self_self_prod_wpaulis = self.wpaulis[:, None].compose_weighted_pauli_array(self.wpaulis[None, :].adjoint())
        prod_paulis, prod_phases = self.paulis[:, None].compose_pauli_array(self.paulis[None, :])
        self_self_docommute_mask = self.wpaulis[:, None].commute_with(self.wpaulis[None, :])

        flat_prod_paulis = prod_paulis.flatten()
        flat_prod_phases = prod_phases.flatten()

        print(prod_paulis.inspect())

        unique_prod_paulis, inverse = pa.fast_flat_unique(flat_prod_paulis, return_inverse=True)

        print(unique_prod_paulis.inspect())
        print(inverse)

        # self_self_mod_2_exp = np.mod(~self_self_docommute_mask + self_phases[:, None] + self_phases[None, :], 2)

        # print("prod pauli")
        # print(self_self_prod_wpaulis[self_self_docommute_mask].inspect())
        # print(self_self_prod_wpaulis[~self_self_docommute_mask].inspect())

        print(self_self_mod_2_exp)
        print(~commute_mask.astype(int))

        three_dee = np.mod(self_self_mod_2_exp[None, :, :] + ~commute_mask[:, :, None] + ~commute_mask[:, None, :], 2)

        print(three_dee)

        # # identifies the unique paulis in the products
        # unique_prod_paulis, inverse = pa.fast_flat_unique(self_self_prod_wpaulis.paulis.flatten(), return_inverse=True)
        # # create a square matrix with the unique pauli index of its position in self_self_prod_wpaulis
        # inverse = inverse.reshape((self.num_terms, self.num_terms))

        # # gathers the weights associated with the unique paulis
        # unique_prod_paulis_weights = np.zeros((unique_prod_paulis.size, unique_prod_paulis.size), dtype=complex)
        # for i in range(unique_prod_paulis.size):
        #     unique_prod_paulis_weights[i, :] = self_self_prod_wpaulis.weights[inverse == i]

        # # identifies the active paulis
        # all_coefs = anticommute_sign @ unique_prod_paulis_weights.T
        # applied_mask = ~np.isclose(all_coefs, 0)
        # active_coefs = np.conj(all_coefs[applied_mask])
        # applied_mask2 = np.mod(
        #     np.arange(applied_mask.size).reshape(all_coefs.shape)[applied_mask],
        #     unique_prod_paulis.size,
        # )
        # active_paulis = unique_prod_paulis[applied_mask2]

        # # performs multiplication between the active paulis and the paulis in other
        # new_paulis, new_phases = active_paulis.compose_pauli_array(flat_other)
        # factors = active_coefs * new_phases

        return new_paulis.reshape(original_shape), factors.reshape(original_shape)

    def expectation_values_from_paulis(self, paulis_expectation_values: NDArray[np.float64]) -> "np.complex":
        """
        Returns the PauliArray expectation value given the expectation values of the Paulis. More useful for other classes, but still here for uniformity.

        Args:
            paulis_expectation_values (NDArray[float]): _description_

        Returns:
            NDArray: _description_
        """
        wpaulis_expectation_values = self.wpaulis.expectation_values_from_paulis(paulis_expectation_values)

        return np.sum(wpaulis_expectation_values)

    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float64]) -> "np.complex":
        """
        Returns the PauliArray expectation value given the expectation values of the Paulis.

        Args:
            paulis_covariances (NDArray[float]): _description_

        Returns:
            NDArray: _description_
        """

        wpaulis_covariances = self.wpaulis.covariances_from_paulis(paulis_covariances)

        return np.sum(wpaulis_covariances)

    def combine_repeated_terms(self, inplace=False) -> Self:
        """
        Combine repeated terms in the sum associated with equal Pauli strings.
        Inspired by : https://github.com/numpy/numpy/issues/11136

        Args:
            inplace (bool, optional): If True, modifies the present instance. Defaults to False.

        Returns:
            Operator: Operator where each Pauli strings appears only once in the sum.
        """

        new_paulis, inverse = pa.fast_flat_unique(self.paulis, return_inverse=True)

        new_weights = np.zeros(new_paulis.shape, dtype=self.wpaulis.weights.dtype)

        np.add.at(new_weights, inverse, self.wpaulis.weights)

        if inplace:
            self._wpaulis = wpa.WeightedPauliArray(new_paulis, new_weights)
            return self

        return Operator.from_paulis_and_weights(new_paulis, new_weights)

    def filter_weights(self, filter_function: callable) -> Self:
        """
        Removes Pauli strings from the Operator object based on the filter_function applied on weights.

        Args:
            filter_function (callable): The filter_function to apply on weights.

        Returns:
            Operator: The filtered Operator.
        """

        return Operator(self.wpaulis.extract(filter_function(self.weights)))

    def remove_small_weights(self, threshold: float = 1e-14) -> Self:
        """
        Remove small weights from the Operator.

        Args:
            threshold (float, optional): The threshold below which weights are considered small. Defaults to 1e-14.

        Returns:
            Operator: The Operator with small weights removed.
        """
        return self.filter_weights(lambda weight: np.abs(weight) > threshold)

    def simplify(self, threshold: float = 1e-14) -> Self:
        """
        Simplify the Operator by removing small weights, combining repeated terms, and again removing small weights.

        Args:
            threshold (float, optional): The threshold below which weights are considered small. Defaults to 1e-14.

        Returns:
            Operator: The simplified Operator.
        """
        return self.remove_small_weights(threshold).combine_repeated_terms().remove_small_weights(threshold)

    def is_scalar(self) -> bool:
        """
        Check if the Operator is a scalar.

        Returns:
            bool: True if the Operator is a scalar.
        """
        return self.num_terms == 1 and np.sum(self.wpaulis[0].paulis.zx_strings) == 0

    def is_unitary(self) -> bool:
        """
        Check if the Operator is unitary.

        Returns:
            bool: True if the Operator is unitary.
        """
        self_prod_wpaulis = self.compose_operator(self.adjoint()).combine_repeated_terms().remove_small_weights()

        return self_prod_wpaulis.is_scalar() and np.isclose(self_prod_wpaulis.wpaulis[0].weights, 1)

    def is_clifford(self) -> bool:
        """
        Check if the Operator is a Clifford operator. For an operator to be Clifford, it first needs to be unitary.
        # TODO : make sure this is robust


        Returns:
            bool: True if the Operator is a Clifford operator.
        """

        sq_amp = np.abs(self.simplify().wpaulis.weights) ** 2

        diff_zx_strings = bitops.row_space(
            bitops.add(self.paulis.zx_strings[:, None, :], self.paulis.zx_strings[None, :, :]).reshape(
                (self.num_terms**2, 2 * self.num_qubits)
            )
        )

        kernel_zx_string = symplectic.orthogonal_complement(diff_zx_strings)

        # print("diffs")
        # print(diff_zx_strings.astype(int))
        # print("kernel")
        # print(kernel_zx_string.astype(int))

        assert np.all(np.isclose(sq_amp, 1 / self.num_terms))

        self_amplitude = 1 / self.num_terms
        self_phases = np.mod(np.round(-np.angle(self.weights) / (np.pi / 2)).astype(int), 4)

        prod_paulis, prod_phases = self.paulis[:, None].compose_pauli_array(self.paulis[None, :])
        self_self_docommute_mask = self.wpaulis[:, None].commute_with(self.wpaulis[None, :])

        flat_prod_paulis = prod_paulis.flatten()
        flat_prod_phases = prod_phases.flatten()

        unique_prod_paulis, inverse = pa.fast_flat_unique(flat_prod_paulis, return_inverse=True)

        # print(flat_prod_paulis.inspect())
        # print(unique_prod_paulis.inspect())
        # print(inverse)

        return True

    def trace(self) -> "np.complex":
        """
        Returns the trace of the Operator.

        Returns:
            "np.complex": The trace of the Operator.
        """
        paulis_traces = self.paulis.traces()

        return np.sum(self.weights * paulis_traces)

    def update_weights(self, new_weights):
        """
        Updates the weights of the Operator with the provided new weights.

        Args:
            new_weights: The new weights to be assigned to the Operator.

        Returns:
            None
        """
        self.wpaulis.update_weights(new_weights)

    def update_weights_from_other(self, other: Self):
        """
        Updates the weights of the Operator from another Operator object.

        Args:
            other (Operator): The Operator from which to update the weights.

        Returns:
            None
        """
        self.wpaulis.update_weights_from_other(other.wpaulis)

    def sort_paulis(self):
        """
        Sorts the underlying WeightedPauliArray primarily for comparison purposes.

        Returns:
            None
        """
        order = pa.argsort(self.paulis)
        self._wpaulis = self.wpaulis[order]

    def to_matrix(self) -> NDArray:
        """
        Converts the Operator into a (n**2, n**2) matrix.

        Returns:
            NDArray: The matrix representation of the Operator.
        """
        # This method does not use WeightedPauliArray.to_matrices() for performance.

        mat_shape = (2**self.num_qubits, 2**self.num_qubits)

        z_ints = bitops.strings_to_ints(self.paulis.z_strings)
        x_ints = bitops.strings_to_ints(self.paulis.x_strings)

        phase_powers = np.mod(bitops.dot(self.paulis.z_strings, self.paulis.x_strings), 4)
        phases = np.choose(phase_powers, [1, -1j, -1, 1j])

        matrix = np.zeros(mat_shape, dtype=complex)
        for idx in np.ndindex(self.wpaulis.shape):
            row_ind, col_ind, matrix_elements = pa.PauliArray.sparse_matrix_from_zx_ints(
                z_ints[idx], x_ints[idx], self.num_qubits
            )
            matrix[row_ind, col_ind] += self.weights[idx] * phases[idx] * matrix_elements

        return matrix

    @classmethod
    def from_labels_and_weights(cls, labels, weights) -> Self:
        """
        Creates an Operator from labels and weights.

        Args:
            labels: The labels of the Operator.
            weights: The weights associated with the labels.

        Returns:
            Operator: The constructed Operator.
        """
        return Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels, weights))

    @classmethod
    def from_paulis_and_weights(cls, paulis, weights) -> Self:
        """
        Creates an Operator from Pauli strings and corresponding weights.

        Args:
            paulis: The Pauli strings.
            weights: The weights associated with the Pauli strings.

        Returns:
            Operator: The constructed Operator.
        """
        return Operator(wpa.WeightedPauliArray(paulis, weights))

    @classmethod
    def from_paulis(cls, paulis) -> Self:
        """
        Creates an Operator from Pauli strings.

        Args:
            paulis: The Pauli strings.

        Returns:
            Operator: The constructed Operator.
        """
        return Operator(wpa.WeightedPauliArray.from_paulis(paulis))

    @classmethod
    def from_matrix(cls, matrix: NDArray, threshold=1e-9):
        """
        Creates an Operator from a matrix representation.

        Args:
            matrix (NDArray): The matrix representation of the Operator.
            threshold (float): Threshold for determining small weights.

        Returns:
            Operator: The constructed Operator.
        """
        num_qubits = int(np.log2(matrix.shape[0]))
        all_paulis = gen_complete_pauli_array_basis(num_qubits)

        weights = np.zeros((all_paulis.size,), dtype=complex)

        all_mats = all_paulis.to_matrices()
        for i in range(all_paulis.size):
            weights[i] = np.trace(all_mats[i, :, :] @ matrix)

        weights *= 1 / (2**num_qubits)

        mask = np.abs(weights) > threshold

        return cls.from_paulis_and_weights(all_paulis[mask], weights[mask])

    @classmethod
    def empty(cls, num_qubits) -> Self:
        """
        Creates an empty Operator with a specified number of qubits.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            Operator: The empty Operator.
        """
        return Operator.from_labels_and_weights(["I" * num_qubits], np.zeros(1))

    @classmethod
    def identity(cls, num_qubits) -> Self:
        """
        Creates an identity Operator with a specified number of qubits.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            Operator: The identity Operator.
        """
        return Operator.from_labels_and_weights(["I" * num_qubits], np.ones(1))

    def to_npz(self, filename):
        """
        Saves the Operator to a .npz file.

        Args:
            filename (str): The name of the .npz file.

        Returns:
            None
        """
        self.wpaulis.to_npz(filename)

    @classmethod
    def from_npz(cls, filename) -> Self:
        """
        Creates an Operator from a .npz file.

        Args:
            filename (str): The name of the .npz file.

        Returns:
            Operator: The constructed Operator.
        """
        return cls(wpa.WeightedPauliArray.from_npz(filename))


def commutator(operator_1: Operator, operator_2: Operator) -> Operator:
    r"""
    Computes the commutator

    .. math::
        [A, B] = AB - BA

    of two Operators.

    Args:
        operator_1 (Operator): The first Operator.
        operator_2 (Operator): The second Operator.

    Returns:
        Operator: The commutator of the two Operators.
    """
    do_commute = operator_1.paulis[:, None].commute_with(operator_2.paulis[None, :])

    idx1, idx2 = np.where(~do_commute)

    new_wpaulis = 2 * operator_1.wpaulis[idx1].compose_weighted_pauli_array(operator_2.wpaulis[idx2])

    new_operator = Operator(new_wpaulis)

    return new_operator


def anticommutator(operator_1: Operator, operator_2: Operator) -> Operator:
    r"""
    Computes the anticommutator

    .. math::
        {A, B} = AB + BA

    of two Operators.

    Args:
        operator_1 (Operator): The first Operator.
        operator_2 (Operator): The second Operator.

    Returns:
        Operator: The anticommutator of the two Operators.
    """
    do_commute = operator_1.paulis[:, None].commute_with(operator_2.paulis[None, :])

    idx1, idx2 = np.where(do_commute)

    new_wpaulis = 2 * operator_1.wpaulis[idx1].compose_weighted_pauli_array(operator_2.wpaulis[idx2])

    new_operator = Operator(new_wpaulis)

    return new_operator
