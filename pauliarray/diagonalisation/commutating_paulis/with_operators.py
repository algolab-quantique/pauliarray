from typing import List, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.operator_array_type_1 as opa
import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import symplectic
from pauliarray.diagonalisation.commutating_paulis.utils import single_qubit_cummuting_generators
from pauliarray.utils.protocols import HasPaulis


def general_to_bitwise(
    paulis: pa.PauliArray, force_single_qubit_generators=False
) -> Tuple[Tuple[pa.PauliArray, NDArray[np.complex128]], opa.OperatorArrayType1]:
    """
    Converts a 1D PauliArray of commuting Pauli strings into bitwise commuting pauli strings and factors. Also returns the transformation which performs the conversion.

    Args:
        paulis (PauliArray): 1D PauliArray of commuting Pauli strings
        force_single_qubit_generators(bool): For already bitwise commuting qubits, the transformation will apply a single qubit rotation to make it diagonal. This prevents some unnecessary n-qubits rotations.

    Returns:
        PauliArray: 1D PauliArray of bitwise commuting Pauli strings
        NDArray[np.complex128]: Phase factors resulting from the transformation
        OperatorArrayType1: The transformation given as a 1D OperatorArrayType1 where the first one is applied first on the original Pauli Array
    """

    assert paulis.ndim == 1
    assert np.all(paulis[:, None].commute_with(paulis[None, :]))

    if force_single_qubit_generators:
        gen_paulis = single_qubit_cummuting_generators(paulis)
        ext_paulis = pa.concatenate((paulis, gen_paulis), axis=0)
        zx_strings = ext_paulis.zx_strings
    else:
        zx_strings = paulis.zx_strings

    lag_zx_strings = symplectic.lagrangian_subspace(zx_strings)
    lag_zx_strings, colag_zx_strings = symplectic.lagrangian_bitwise_colagrangian_subspaces(lag_zx_strings)

    commuting_generators = pa.PauliArray.from_zx_strings(lag_zx_strings)
    conjugate_generators = pa.PauliArray.from_zx_strings(colag_zx_strings)

    commuting_operators = opa.OperatorArrayType1.from_pauli_array(commuting_generators)
    conjugate_operators = opa.OperatorArrayType1.from_pauli_array(conjugate_generators)

    transformations = commuting_operators.add_operator_array_type_1(conjugate_operators).mul_weights(np.sqrt(0.5))

    new_paulis, factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

    assert np.all(new_paulis[:, None].bitwise_commute_with(new_paulis[None, :]))

    return new_paulis, factors, transformations


def bitwise_to_diagonal(
    paulis: pa.PauliArray,
) -> Tuple[Tuple[pa.PauliArray, NDArray[np.complex128]], opa.OperatorArrayType1]:
    """
    Converts a 1D PauliArray of bitwise commuting Pauli strings into diagonal commuting pauli strings and factors. Also returns the transformation which performs the conversion.

    Args:
        paulis (pa.PauliArray): 1D PauliArray of bitwise commuting Pauli strings

    Returns:
        PauliArray: 1D PauliArray of diagonal Pauli strings
        NDArray[np.complex128]: Phase factors resulting from the transformation
        OperatorArrayType1: The transformation given as a 1D OperatorArrayType1 where the first one is applied first on the original Pauli Array
    """

    assert paulis.ndim == 1
    assert np.all(paulis[:, None].bitwise_commute_with(paulis[None, :]))

    num_qubits = paulis.num_qubits

    all_positions = np.arange(num_qubits)
    x_positions = all_positions[
        np.any(
            np.logical_and(paulis.x_strings, ~paulis.z_strings),
            axis=tuple(range(paulis.ndim)),
        )
    ]
    y_positions = all_positions[
        np.any(
            np.logical_and(paulis.x_strings, paulis.z_strings),
            axis=tuple(range(paulis.ndim)),
        )
    ]

    num_transformations = len(x_positions) + len(y_positions)

    commuting_z_strings = np.zeros((num_transformations, num_qubits), dtype=bool)
    commuting_x_strings = np.zeros((num_transformations, num_qubits), dtype=bool)

    conjugate_z_strings = np.zeros((num_transformations, num_qubits), dtype=bool)
    conjugate_x_strings = np.zeros((num_transformations, num_qubits), dtype=bool)

    i = 0
    for x_position in x_positions:
        commuting_x_strings[i, x_position] = True
        conjugate_z_strings[i, x_position] = True
        i += 1

    for y_position in y_positions:
        commuting_x_strings[i, y_position] = True
        commuting_z_strings[i, y_position] = True
        conjugate_z_strings[i, y_position] = True
        i += 1

    commuting_generators = pa.PauliArray(commuting_z_strings, commuting_x_strings)
    conjugate_generators = pa.PauliArray(conjugate_z_strings, conjugate_x_strings)

    commuting_operators = opa.OperatorArrayType1.from_pauli_array(commuting_generators)
    conjugate_operators = opa.OperatorArrayType1.from_pauli_array(conjugate_generators)

    transformations = commuting_operators.add_operator_array_type_1(conjugate_operators).mul_weights(np.sqrt(0.5))

    new_paulis, factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

    assert np.all(new_paulis.is_diagonal())

    return new_paulis, factors, transformations


def general_to_diagonal(
    paulis: pa.PauliArray, force_single_qubit_generators=False
) -> Tuple[Tuple[pa.PauliArray, NDArray[np.complex128]], opa.OperatorArrayType1]:
    """
    Converts a 1D PauliArray of commuting Pauli strings into diagonal commuting pauli strings and factors. Also returns the transformation which performs the conversion.

    Args:
        paulis (pa.PauliArray): 1D PauliArray of commuting Pauli strings
        force_single_qubit_generators(bool): For already bitwise commuting qubits, the transformation will apply a single qubit rotation to make it diagonal. This prevents some unnecessary n-qubits rotations.

    Returns:
        PauliArray: 1D PauliArray of diagonal Pauli strings
        NDArray[np.complex128]: Phase factors resulting from the transformation
        OperatorArrayType1: The transformation given as a 1D OperatorArrayType1 where the first one is applied first on the original Pauli Array
    """

    bitwise_paulis, g_to_b_factors, general_to_bitwise_ops = general_to_bitwise(
        paulis, force_single_qubit_generators=force_single_qubit_generators
    )
    diagonal_paulis, b_to_d_add_factors, bitwise_to_diagonal_ops = bitwise_to_diagonal(bitwise_paulis)

    g_to_d_factors = g_to_b_factors * b_to_d_add_factors

    transformations = opa.concatenate((general_to_bitwise_ops, bitwise_to_diagonal_ops), axis=0)

    return (diagonal_paulis, g_to_d_factors), transformations


def diagonalise_with_operators(pauli_obj: HasPaulis, force_single_qubit_generators=False) -> Union[
    Tuple[Tuple[pa.PauliArray, NDArray[np.complex128]], opa.OperatorArrayType1],
    Tuple[HasPaulis, opa.OperatorArrayType1],
]:

    paulis = pauli_obj.paulis

    flat_paulis = paulis.flatten()

    (flat_diagonal_paulis, flat_factors), transformations = general_to_diagonal(
        flat_paulis, force_single_qubit_generators=force_single_qubit_generators
    )

    diagonal_paulis = flat_diagonal_paulis.reshape(paulis.shape)
    factors = flat_factors.reshape(paulis.shape)

    diagonal_pauli_obj = pauli_obj.replace_paulis(diagonal_paulis).mul_weights(factors)

    return (diagonal_pauli_obj, transformations)
