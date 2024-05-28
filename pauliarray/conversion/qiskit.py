from typing import List

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_nature.second_q.operators.fermionic_op import FermionicOp

from pauliarray import Operator, OperatorArrayType1, PauliArray, WeightedPauliArray


def pauli_array_to_pauli_list(paulis: PauliArray) -> PauliList:
    """
    Converts a PauliArray to a Qiskit's PauliList.

    Args:
        paulis (PauliArray): A PauliArray object.

    Returns:
        PauliList: The corresponding PauliList object.
    """
    assert paulis.ndim == 1
    return PauliList.from_symplectic(z=paulis.z_strings, x=paulis.x_strings)


def operator_to_sparse_pauli(
    operator: Operator,
) -> SparsePauliOp:
    """
    Converts an Operator to a Qiskit's SparsePauliOp.

    Args:
        operator (Operator): An Operator object.

    Returns:
        SparsePauliOp: The corresponding SparsePauliOp object.
    """
    return SparsePauliOp(pauli_array_to_pauli_list(operator.paulis), coeffs=operator.weights)


def pauli_array_from_pauli_list(pauli_list: PauliList) -> PauliArray:
    """
    Converts a Qiskit's PauliList to a PauliArray.

    Args:
        pauli_list (PauliList): A PauliList to convert.

    Returns:
        PauliArray: The corresponding PauliArray object.
    """
    return PauliArray(pauli_list.z, pauli_list.x)


def weighted_pauli_array_from_pauli_list(
    pauli_list: PauliList,
) -> WeightedPauliArray:
    """
    Converts a Qiskit's PauliList to a WeightedPauliArray.

    Args:
        pauli_list (PauliList): A PauliList to convert.

    Returns:
        WeightedPauliArray: The corresponding WeightedPauliArray object.
    """
    return WeightedPauliArray.from_z_strings_and_x_strings_and_weights(
        pauli_list.z,
        pauli_list.x,
        (-1j) ** pauli_list.phase,
    )


def operator_from_sparse_pauli(sparse_pauli: SparsePauliOp) -> Operator:
    """
    Converts a Qiskit's SparsePauliOp to an Operator.

    Args:
        sparse_pauli (SparsePauliOp): A SparsePauliOp.

    Returns:
        Operator: The corresponding Operator object.
    """
    return Operator.from_paulis_and_weights(pauli_array_from_pauli_list(sparse_pauli.paulis), sparse_pauli.coeffs)


def operator_array_to_sparse_pauli_list(operator_array: OperatorArrayType1) -> list[SparsePauliOp]:
    """
    Converts an OperatorArrayType1 to a list of Qiskit's SparsePauliOps.

    Args:
        operator_array (OperatorArrayType1): An OperatorArray.

    Returns:
        list[SparsePauliOp]: A list of corresponding SparsePauliOp objects.
    """
    return [operator_to_sparse_pauli(Operator(op.wpaulis)) for op in operator_array]


def operator_array_from_sparse_pauli_list(sparse_paulis: List[SparsePauliOp]) -> OperatorArrayType1:
    """
    Converts a list of Qiskit's SparsePauliOps to an OperatorArrayType1.

    Args:
        sparse_paulis (List[SparsePauliOp]): A list of SparsePauliOp objects.

    Returns:
        OperatorArrayType1: An OperatorArray object.
    """
    operator_list = [operator_from_sparse_pauli(sparse_pauli) for sparse_pauli in sparse_paulis]

    return OperatorArrayType1.from_operator_list(operator_list)


def extract_fermionic_op(
    fermionic_op: FermionicOp,
) -> tuple[
    tuple[list["np.ndarray[np.int]"], NDArray, list[list]], tuple[list["np.ndarray[np.int]"], NDArray, list[list]]
]:
    """
    Extracts data from a Qiskit Nature FermionicOp to be used by PauliArray mapping.

    Args:
        fermionic_op (FermionicOp): A FermionicOp object.

    Returns:
        tuple[tuple[list["np.ndarray[np.int]"], NDArray, list[list]], tuple[list["np.ndarray[np.int]"], NDArray, list[list]]]:
        Two tuples containing one-body and two-body terms.
            Each tuple consists of:
            - list["np.ndarray[np.int]"]: Orbital indices for the terms.
            - NDArray: Values of the terms.
            - list[list]: Signs of the terms.
    """
    one_body_orbitals = [[], []]
    one_body_values = []
    one_body_signs = [[], []]

    two_body_orbitals = [[], [], [], []]
    two_body_values = []
    two_body_signs = [[], [], [], []]

    SIGNS = {"+": 1, "-": -1}

    for term in fermionic_op.terms():
        if len(term[0]) == 2:
            for index in range(2):
                one_body_orbitals[index].append(term[0][index][1])
                one_body_signs[index].append(SIGNS[term[0][index][0]])
            one_body_values.append(term[1])
        elif len(term[0]) == 4:
            for index in range(4):
                two_body_orbitals[index].append(term[0][index][1])
                two_body_signs[index].append(SIGNS[term[0][index][0]])
            two_body_values.append(term[1])
        else:
            raise ValueError("Fermionic operators must count 2 or 4 creation/annihilation operators.")

    one_body_orbitals = [np.array(c, dtype=np.int_) for c in one_body_orbitals]
    two_body_orbitals = [np.array(c, dtype=np.int_) for c in two_body_orbitals]
    one_body_values = np.array(one_body_values)
    two_body_values = np.array(two_body_values)

    one_body_tuple = (one_body_orbitals, one_body_values, one_body_signs)
    two_body_tuple = (two_body_orbitals, two_body_values, two_body_signs)

    return one_body_tuple, two_body_tuple
