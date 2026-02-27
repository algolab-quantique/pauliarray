from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops
from pauliarray.binary.matrix_library import build_heavyside_matrix, build_parity_matrix


def assemble_real_imag_majoranas(mapping_matrix: NDArray[np.bool_]) -> Tuple[pa.PauliArray, pa.PauliArray]:
    r"""
    In a majorana-to-pauli strings mapping, each real/imag majorana operator is a Pauli string. This function construct these majorana operators.

    Returns:
        PauliArray: The Pauli strings for :math:`P_\text{real}`
        PauliArray: The Pauli strings for :math:`P_\text{imag}`
    """

    assert mapping_matrix.ndim == 2, "Mapping Matrix must be square"
    assert mapping_matrix.shape[0] == mapping_matrix.shape[1], "Mapping Matrix must be square"

    num_qubits = mapping_matrix.shape[0]

    mapping_matrix_inv = bitops.inv(mapping_matrix)

    heavyside_matrix = build_heavyside_matrix(num_qubits)
    parity_matrix = build_parity_matrix(num_qubits)

    real_z_strings = bitops.matmul(heavyside_matrix, mapping_matrix_inv)
    imag_z_strings = bitops.matmul(parity_matrix, mapping_matrix_inv)
    real_x_strings = imag_x_strings = mapping_matrix.transpose()

    real_majoranas = pa.PauliArray(real_z_strings, real_x_strings)
    imag_majoranas = pa.PauliArray(imag_z_strings, imag_x_strings)

    return real_majoranas, imag_majoranas
