from typing import Union

import numpy as np
from numpy.typing import NDArray

from pauliarray.pauli.pauli_array import PauliArray
from pauliarray.utils.decorators import transform_paulis_with_clifford


@transform_paulis_with_clifford
def h(paulis: PauliArray, qubits: Union[int, list[int]]) -> tuple[PauliArray, NDArray]:
    """
    Performs a Clifford conjugaison by H on given qubits. This exchanges X for Z and vice-versa and Y into -Y.

    Args:
        qubits (int or list[int]): The qubits on which to apply H.
        inplace (bool): Apply the changes to self if True. Return a modified copy if False.

    Returns:
        PauliArray: The transformed PauliArray
        "np.ndarray[np.complex]": The factors resulting from the transformation
    """

    if isinstance(qubits, int):
        qubits = [qubits]

    y_strings = np.logical_and(paulis.x_strings, paulis.z_strings)
    add_phases = 2 * np.mod(np.sum(y_strings[..., qubits], axis=-1), 2)

    new_z_strings = paulis.z_strings.copy()
    new_x_strings = paulis.x_strings.copy()
    new_quad_phases = np.mod(add_phases, 4)

    new_z_strings[..., qubits], new_x_strings[..., qubits] = (
        new_x_strings[..., qubits],
        new_z_strings[..., qubits],
    )

    factors = np.choose(new_quad_phases, [1, 1j, -1, -1j])

    return PauliArray(new_z_strings, new_x_strings), factors


@transform_paulis_with_clifford
def s(paulis: PauliArray, qubits: Union[int, list[int]]) -> tuple[PauliArray, NDArray]:
    """
    Performs a Clifford conjugaison by S on given qubits. With this transformation X -> Y and Y -> -X.

    Args:
        qubits (int or list[int]): The qubits on which to apply S.
        inplace (bool): Apply the changes to self if True. Return a modified copy if False.

    Returns:
        PauliArray: The transformed PauliArray
        "np.ndarray[np.complex]": The factors resulting from the transformation
    """

    if isinstance(qubits, int):
        qubits = [qubits]

    y_strings = np.logical_and(paulis.x_strings, paulis.z_strings)
    add_phases = 2 * np.sum(y_strings[..., qubits], axis=-1)

    new_z_strings = paulis.z_strings.copy()
    new_x_strings = paulis.x_strings.copy()
    new_quad_phases = np.mod(add_phases, 4)

    new_z_strings[..., qubits] = np.logical_xor(paulis.z_strings[..., qubits], paulis.x_strings[..., qubits])

    factors = np.choose(new_quad_phases, [1, 1j, -1, -1j])

    return PauliArray(new_z_strings, new_x_strings), factors


@transform_paulis_with_clifford
def cx(
    paulis: PauliArray, control_qubits: Union[int, list[int]], target_qubits: Union[int, list[int]]
) -> tuple[PauliArray, NDArray]:
    """
    Performs a Clifford conjugaison by CX on given qubits. If multiple control and target qubits are given, the gates are apply in the given order.

    Args:
        control_qubits (int or list[int]): The control qubits on which of the CX.
        target_qubits (int or list[int]): The target qubits on which of the CX.
        inplace (bool): Apply the changes to self if True. Return a modified copy if False.

    Returns:
        PauliArray: The transformed PauliArray
        "np.ndarray[np.complex]": The factors resulting from the transformation
    """

    if isinstance(control_qubits, int):
        control_qubits = [control_qubits]
    if isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    assert len(control_qubits) == len(target_qubits)

    new_z_strings = paulis.z_strings.copy()
    new_x_strings = paulis.x_strings.copy()

    add_phases = np.zeros(paulis.shape, dtype=int)
    for cq, tq in zip(control_qubits, target_qubits):
        add_phases += (
            2
            * new_x_strings[..., cq]
            * new_z_strings[..., tq]
            * np.logical_not(np.logical_xor(new_z_strings[..., cq], new_x_strings[..., tq]))
        )

        tmp_tq_x_bit_array = new_x_strings[..., tq].copy()
        tmp_cq_z_bit_array = new_z_strings[..., cq].copy()
        new_x_strings[..., tq] = np.logical_xor(tmp_tq_x_bit_array, new_x_strings[..., cq])
        new_z_strings[..., cq] = np.logical_xor(tmp_cq_z_bit_array, new_z_strings[..., tq])

    new_quad_phases = np.mod(add_phases, 4)
    factors = np.choose(new_quad_phases, [1, 1j, -1, -1j])

    return PauliArray(new_z_strings, new_x_strings), factors


@transform_paulis_with_clifford
def cz(
    paulis: PauliArray, control_qubits: Union[int, list[int]], target_qubits: Union[int, list[int]]
) -> tuple[PauliArray, NDArray]:
    """
    Performs a Clifford conjugaison by CZ on given qubits. If multiple control and target qubits are given, the gates are apply in the given order.

    Args:
        control_qubits (int or list[int]): The control qubits on which of the CZ.
        target_qubits (int or list[int]): The target qubits on which of the CZ.
        inplace (bool): Apply the changes to self if True. Return a modified copy if False.

    Returns:
        PauliArray: The transformed PauliArray
        "np.ndarray[np.complex]": The factors resulting from the transformation
    """

    if isinstance(control_qubits, int):
        control_qubits = [control_qubits]
    if isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    assert len(control_qubits) == len(target_qubits)

    new_z_strings = paulis.z_strings.copy()
    new_x_strings = paulis.x_strings.copy()

    add_phases = np.zeros(paulis.shape, dtype=int)
    for cq, tq in zip(control_qubits, target_qubits):
        add_phases += (
            2
            * new_x_strings[..., cq]
            * new_x_strings[..., tq]
            * np.logical_xor(new_z_strings[..., cq], new_z_strings[..., tq])
        )

        new_z_strings[..., cq] = np.logical_xor(new_z_strings[..., cq], new_x_strings[..., tq])
        new_z_strings[..., tq] = np.logical_xor(new_x_strings[..., cq], new_z_strings[..., tq])

    new_quad_phases = np.mod(add_phases, 4)
    factors = np.choose(new_quad_phases, [1, 1j, -1, -1j])

    return PauliArray(new_z_strings, new_x_strings), factors
