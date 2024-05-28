import numpy as np
from numpy.typing import NDArray


def bit_sum(bit_strings: "np.ndarray[np.bool]") -> int:
    """
    Calculates the sum of booleans along the last axis of the input array.

    Args:
        bit_strings ("np.ndarray[np.bool]"): Array of booleans.

    Returns:
        int: Sum of bits along the last axis.
    """
    return np.sum(bit_strings, axis=-1)


def dot(
    bit_strings_1: "np.ndarray[np.bool]",
    bit_strings_2: "np.ndarray[np.bool]",
) -> "np.ndarray[np.int]":
    """
    Computes the dot product between two arrays of boolean values.

    Args:
        bit_strings_1 ("np.ndarray[np.bool]"): First array of boolean values.
        bit_strings_2 ("np.ndarray[np.bool]"): Second array of boolean values.

    Returns:
        "np.ndarray[np.int]": Dot product of the two input arrays.
    """
    return bit_sum(bit_strings_1 * bit_strings_2)


def matmul(
    bit_matrix_1: "np.ndarray[np.bool]",
    bit_matrix_2: "np.ndarray[np.bool]",
) -> "np.ndarray[np.bool]":
    """
    Performs matrix multiplication over binary matrices.

    Args:
        bit_matrix_1 ("np.ndarray[np.bool]"): First binary matrix.
        bit_matrix_2 ("np.ndarray[np.bool]"): Second binary matrix.

    Returns:
        "np.ndarray[np.bool]": Resultant binary matrix after multiplication.
    """
    return np.mod(np.matmul(bit_matrix_1.astype(int), bit_matrix_2.astype(int)), 2).astype(bool)


def add(
    bit_matrix_1: "np.ndarray[np.bool]",
    bit_matrix_2: "np.ndarray[np.bool]",
) -> "np.ndarray[np.bool]":
    """
    Performs element-wise XOR operation between two binary matrices.

    Args:
        bit_matrix_1 ("np.ndarray[np.bool]"): First binary matrix.
        bit_matrix_2 ("np.ndarray[np.bool]"): Second binary matrix.

    Returns:
        "np.ndarray[np.bool]": Element-wise XOR result of the two input matrices.
    """
    return np.logical_xor(bit_matrix_1, bit_matrix_2)


def rank(bit_matrix: "np.ndarray[np.bool]") -> int:
    """
    Computes the rank of a binary matrix.

    Args:
        bit_matrix ("np.ndarray[np.bool]"): Input binary matrix.

    Returns:
        int: Rank of the binary matrix.
    """
    assert bit_matrix.ndim == 2

    row_ech = row_space(bit_matrix)

    return row_ech.shape[0]


def inv(bit_matrix: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Computes the inverse of a binary matrix.

    Args:
        bit_matrix ("np.ndarray[np.bool]"): Input binary matrix.

    Returns:
        "np.ndarray[np.bool]": Inverse of the input binary matrix.
    """
    assert bit_matrix.ndim == 2

    return np.linalg.inv(bit_matrix.astype(np.int8)).astype(bool)


def strings_to_ints(bit_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.int]":
    """
    Converts binary strings to integers.

    Args:
        bit_strings ("np.ndarray[np.bool]"): Input array of binary strings.

    Returns:
        "np.ndarray[np.int]": Integers obtained from input binary strings
    """
    power_of_twos = 1 << np.arange(bit_strings.shape[-1])

    return bit_sum(bit_strings * power_of_twos)


def row_echelon(bit_matrix: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Applies Gauss-Jordan elimination on a binary matrix to produce row echelon form.

    Args:
        bit_matrix ("np.ndarray[np.bool]"): Input binary matrix.

    Returns:
        "np.ndarray[np.bool]": Row echelon form of the provided matrix.
    """
    re_bit_matrix = bit_matrix.copy()

    n_rows = re_bit_matrix.shape[0]
    n_cols = re_bit_matrix.shape[1]

    row_range = np.arange(n_rows)

    h_row = 0
    k_col = 0

    while h_row < n_rows and k_col < n_cols:
        if np.all(re_bit_matrix[h_row:, k_col] == 0):
            k_col += 1
        else:
            i_row = h_row + np.argmax(re_bit_matrix[h_row:, k_col])
            if i_row != h_row:
                re_bit_matrix[[i_row, h_row], :] = re_bit_matrix[[h_row, i_row], :]

            cond_rows = np.logical_and(re_bit_matrix[:, k_col], (row_range != h_row))

            re_bit_matrix[cond_rows, :] = np.logical_xor(re_bit_matrix[cond_rows, :], re_bit_matrix[h_row, :][None, :])

            h_row += 1
            k_col += 1

    return re_bit_matrix


def kernel(bit_matrix: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    r"""
    Computes the Kernel of a two dimensions "np.ndarray[np.bool]".
    The Kernel of a matrix A is a matrix which the columns are formed by the vectors x such that

    .. math::
        A x = 0.

    Args:
        bit_matrix ("np.ndarray[np.bool]"): Input binary matrix.

    Returns:
        "np.ndarray[np.bool]": Kernel of the input binary matrix.
    """
    assert bit_matrix.ndim == 2
    assert bit_matrix.dtype == np.dtype(bool)

    re_bit_matrix = bit_matrix.T

    n_rows = re_bit_matrix.shape[0]
    n_cols = re_bit_matrix.shape[1]

    ext_bit_matrix = np.concatenate([re_bit_matrix, np.eye(n_rows, dtype=bool)], axis=1)

    row_ech_ext_bit_matrix = row_echelon(ext_bit_matrix)

    row_ech_bit_matrix = row_ech_ext_bit_matrix[:, :n_cols]
    inverse_bit_matrix = row_ech_ext_bit_matrix[:, n_cols:]

    null_rows = np.all(~row_ech_bit_matrix, axis=1)

    return inverse_bit_matrix[null_rows, :]


def intersection_row_space(
    bit_matrix_1: "np.ndarray[np.bool]", bit_matrix_2: "np.ndarray[np.bool]"
) -> "np.ndarray[np.bool]":
    """
    Given two matrices which rows are spanning two subspace, this fonction returns rows spanning the intersection
    subspace.

    Args:
        bit_matrix_1 ("np.ndarray[np.bool]"): First binary matrix.
        bit_matrix_2 ("np.ndarray[np.bool]"): Second binary matrix.

    Returns:
        "np.ndarray[np.bool]": Rows spanning the intersection subspace.
    """
    assert bit_matrix_1.ndim == bit_matrix_2.ndim == 2
    assert bit_matrix_1.shape[1] == bit_matrix_2.shape[1]

    rs_bit_matrix_1 = row_space(bit_matrix_1)
    rs_bit_matrix_2 = row_space(bit_matrix_2)

    num_rows_1 = rs_bit_matrix_1.shape[0]

    all_rows = np.concatenate((rs_bit_matrix_1, rs_bit_matrix_2), 0)

    null_row_combination = kernel(all_rows.T)

    return matmul(null_row_combination[:, :num_rows_1], all_rows[:num_rows_1, :])


def row_space(bits: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Computes the row space of a binary matrix using Gauss-Jordan elimination and
    removing the zero lines.

    Args:
        bits ("np.ndarray[np.bool]"): Input binary matrix.

    Returns
        "np.ndarray[np.bool]": Row space of the current matrix
    """
    row_ech_bits = row_echelon(bits)
    null_rows = np.all(~row_ech_bits, axis=1)

    return row_ech_bits[~null_rows, :]


def orthogonal_basis(bit_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Computes the orthogonal basis of the given set of bit strings.

    Args:
        bit_strings ("np.ndarray[np.bool]"): Input array of bit strings.

    Returns
        "np.ndarray[np.bool]": Orthogonal basis of the input bit strings.
    """
    return row_space(bit_strings)


def orthogonal_complement(bit_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Computes the orthogonal complement of the given set of bit strings.

    Args:
        bit_strings ("np.ndarray[np.bool]"): Input array of bit strings.

    Returns
        "np.ndarray[np.bool]": Orthogonal basis of the input bit strings.
    """
    return kernel(bit_strings)


def intersection(bit_strings_1: "np.ndarray[np.bool]", bit_strings_2: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Computes the intersection of two sets of bit strings.

    Args
        bit_strings_1 ("np.ndarray[np.bool]"): First array of boolean values.
        bit_strings_2 ("np.ndarray[np.bool]"): Second array of boolean values.

    Returns
        "np.ndarray[np.bool]": Intersection of the input bit strings.
    """
    return intersection_row_space(bit_strings_1, bit_strings_2)


def is_orthogonal(bit_strings_1: "np.ndarray[np.bool]", bit_strings_2: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Checks if two sets of bit strings are orthogonal.

    Args:
        bit_strings_1 ("np.ndarray[np.bool]"): First array of boolean values.
        bit_strings_2 ("np.ndarray[np.bool]"): Second array of boolean values.

    Returns
        "np.ndarray[np.bool]": True if the input bit strings are orthogonal, False otherwise.
    """
    assert bit_strings_1.shape[-1] == bit_strings_2.shape[-1]
    assert bit_strings_1.shape[-1] % 2 == 0

    return ~dot(bit_strings_1, bit_strings_2)
