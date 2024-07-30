from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.operator_array_type_1 as opa
import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import symplectic


def general_to_bitwise(
    paulis: pa.PauliArray,
) -> Tuple[pa.PauliArray, NDArray[np.complex_], opa.OperatorArrayType1]:
    """
    Converts a PauliArray of commuting Pauli strings into bitwise commuting pauli strings and factors. Also returns the transformation which performs the conversion.

    Args:
        paulis (pa.PauliArray): _description_

    Returns:
        Tuple[pa.PauliArray, NDArray[np.complex_], opa.OperatorArrayType1]: _description_
    """

    assert paulis.ndim == 1
    assert np.all(paulis[:, None].commute_with(paulis[None, :]))

    lag_zx_strings = symplectic.lagrangian_subspace(paulis.zx_strings)
    colag_zx_strings = symplectic.conjugate_subspace(lag_zx_strings)

    lag_zx_strings, colag_zx_strings = symplectic.simplify_lagrangian_colagrangian(lag_zx_strings, colag_zx_strings)

    commuting_generators = pa.PauliArray(*symplectic.split_zx_strings(lag_zx_strings))
    conjugate_generators = pa.PauliArray(*symplectic.split_zx_strings(colag_zx_strings))

    commuting_operators = opa.OperatorArrayType1.from_pauli_array(commuting_generators)
    conjugate_operators = opa.OperatorArrayType1.from_pauli_array(conjugate_generators)

    transformations = np.sqrt(0.5) * commuting_operators.add_operator_array_type_1(conjugate_operators)

    new_paulis, factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

    assert np.all(new_paulis[:, None].bitwise_commute_with(new_paulis[None, :]))

    return new_paulis, factors, transformations


def bitwise_to_diagonal(
    paulis: pa.PauliArray,
) -> Tuple[pa.PauliArray, NDArray[np.complex_], opa.OperatorArrayType1]:
    """
    Converts a PauliArray of bitwise commuting Pauli strings into diagonal commuting pauli strings and factors. Also returns the transformation which performs the conversion.

    Args:
        paulis (pa.PauliArray): _description_

    Returns:
        Tuple[pa.PauliArray, NDArray[np.complex_], opa.OperatorArrayType1]: _description_
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

    transformations = np.sqrt(0.5) * commuting_operators.add_operator_array_type_1(conjugate_operators)

    new_paulis, factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

    assert np.all(new_paulis.is_diagonal())

    return new_paulis, factors, transformations


def general_to_diagonal(
    paulis: pa.PauliArray,
) -> Tuple[pa.PauliArray, NDArray[np.complex_], opa.OperatorArrayType1]:

    bitwise_paulis, factors, general_to_bitwise_ops = general_to_bitwise(paulis)
    diagonal_paulis, add_factors, bitwise_to_diagonal_ops = bitwise_to_diagonal(bitwise_paulis)

    factors *= add_factors

    transformations = opa.concatenate((general_to_bitwise_ops, bitwise_to_diagonal_ops), axis=0)

    return diagonal_paulis, factors, transformations
