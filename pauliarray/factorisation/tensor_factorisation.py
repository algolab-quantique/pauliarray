import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.operator_array_type_1 as opua
import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import symplectic

# from pauliarray.state.basis_state_array import BasisStateArray


def find_symmetry_paulis(paulis: pa.PauliArray) -> pa.PauliArray:
    """
    Return a PauliArray from which each Pauli string commutes with all the Pauli strings in `paulis`.

    Args:
        paulis (pa.PauliArray): A set of Pauli string

    Returns:
        pa.PauliArray: The Pauli strings which commutes with the provided Pauli strings.
    """
    ortho_zx_strings = symplectic.orthogonal_complement(paulis.zx_strings)

    return pa.PauliArray(*symplectic.split_zx_strings(ortho_zx_strings))


def identify_factor_qubits(paulis: pa.PauliArray) -> NDArray[np.bool_]:
    """
    Factor qubits for a set of Pauli strings are qubits which are act on only by the identity and at most one Pauli operator. These qubits can be replaced by the eigenvalue of the corresponding Pauli operator or identity.

    Args:
        paulis (pa.PauliArray): A set of Pauli string

    Returns:
        NDArray[np.bool_]: Mask identifying the factor qubits
    """
    num_qubits = paulis.num_qubits

    factor_qubits = np.zeros((num_qubits,), dtype=bool)
    for i in range(num_qubits):
        unique_paulis = pa.unique(paulis.take_qubits(i))

        if np.sum(np.any(unique_paulis.zx_strings, axis=1).astype(int)) < 2:  # Maximum of one Pauli
            factor_qubits[i] = True

    return factor_qubits


def identify_trivial_qubits(paulis: pa.PauliArray):
    num_qubits = paulis.num_qubits

    trivial_qubits = np.zeros((num_qubits,), dtype=bool)
    for i in range(num_qubits):
        unique_paulis = pa.unique(paulis.take_qubits(i))

        if np.sum(np.any(unique_paulis.zx_strings.bit_array.bits, axis=1).astype(int)) < 1:  # Maximum non Pauli
            trivial_qubits[i] = True

    return trivial_qubits


def symmetries_to_qubit_transformations(symmetry_paulis: pa.PauliArray) -> opua.OperatorArrayType1:
    """
    Given commutating symmetries given as Pauli strings, constructs a series of transformation which brings each symmetry to a single qubit.

    Args:
        symmetry_paulis (pa.PauliArray): _description_

    Returns:
        opua.OperatorArrayType1: _description_
    """

    assert symmetry_paulis.ndim == 1
    assert np.all(symmetry_paulis[:, None].commute_with(symmetry_paulis[None, :]))

    conjugate_paulis = pa.PauliArray.from_zx_strings(symplectic.conjugate_subspace(symmetry_paulis.zx_strings))

    symmetry_operators = opua.OperatorArrayType1.from_pauli_array(symmetry_paulis)
    conjugate_operators = opua.OperatorArrayType1.from_pauli_array(conjugate_paulis)

    transformations = np.sqrt(0.5) * symmetry_operators.add_operator_array_type_1(conjugate_operators)

    return transformations


# def remove_qubits_with_pauli_symmetries(
#     wpaulis: wpa.WeightedPauliArray, symmetry_paulis: pa.PauliArray, symmetry_eigvalues: NDArray[np.int_]
# ) -> wpa.WeightedPauliArray:
#     transformations = symmetries_to_qubit_transformations(symmetry_paulis)

#     mod_wpaulis = wpaulis.copy()
#     for i in range(transformations.size):
#         mod_wpaulis.clifford_conjugate(transformations.get_operator(i), inplace=True)

#     factor_qubits = identify_factor_qubits(mod_wpaulis.paulis)
#     factor_paulis = mod_wpaulis.paulis.compress_qubits(factor_qubits)

#     factor_state = BasisStateArray.from_computationnal_eigvalues(symmetry_eigvalues)
#     factor_eigvalues = factor_state.pauli_array_expectation_value(factor_paulis.flip_zx())

#     new_wpaulis = mod_wpaulis.compress_qubits(~factor_qubits).mul_weights(factor_eigvalues)

#     return new_wpaulis
