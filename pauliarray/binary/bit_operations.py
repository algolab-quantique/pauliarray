from typing import Tuple, Union

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

    return np.linalg.inv(bit_matrix.astype(np.uint8)).astype(bool)


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


def fast_flat_unique_bit_string(
    bit_strings: NDArray[np.bool_],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Union[NDArray[np.bool_], Tuple[NDArray[np.bool_], NDArray]]:
    """
    Faster version of unique for bit string. Only works with flat a list of bit strings (2d array).
    Directly uses numpy.unique.

    Args:
        bit_strings (PauliArray): List of bit strings. Must be 2d and the second dimension along the length of the bit strings.

        return_index (bool, optional): If True, also return the indices of PauliArray (along the specified axis,
            if provided, or in the flattened array) that result in the unique array. Defaults to False.

        return_inverse (bool, optional): If True, also return the indices of the unique array
            (for the specified axis, if provided) that can be used to reconstruct array. Defaults to False.

        return_counts (bool, optional): If True, also return the number of times each unique item appears in array.
            Defaults to False.

    Returns:
        NDArray: The unique bit strings
        NDArray, optional: Index to get unique from the orginal PauliArray
        NDArray, optional: Innverse to reconstrut the original PauliArray from unique
        NDArray, optional: The number of each unique in the original PauliArray
    """

    assert bit_strings.ndim == 2

    void_type_size = bit_strings.dtype.itemsize * bit_strings.shape[-1]

    string_view = np.ascontiguousarray(bit_strings).view(np.dtype((np.void, void_type_size)))

    _, index, inverse, counts = np.unique(string_view, return_index=True, return_inverse=True, return_counts=True)

    new_bitstring = bit_strings[index, :]

    out = (new_bitstring,)
    if return_index:
        out += (index,)
    if return_inverse:
        out += (inverse,)
    if return_counts:
        out += (counts,)

    if len(out) == 1:
        return out[0]

    return out


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


def pack_diagonal(
    bit_strings: "np.ndarray[np.bool]", start_index: int = 0
) -> Tuple["np.ndarray[np.bool]", "np.ndarray[np.bool]", "np.ndarray[np.int]", int]:
    """
    Apply row operations, and column reordering to place "1" on the diagonal starting from the "start_index" diagonal element.

    Returns:
        _type_: _description_
    """

    bit_strings = bit_strings.copy()

    num_rows, num_cols = bit_strings.shape

    row_range = np.arange(num_rows)
    col_order = np.arange(num_cols)
    row_op = np.eye(num_rows)

    while start_index < num_rows:
        failed = True
        for hot_col in range(start_index, num_cols):
            if np.any(bit_strings[start_index:, hot_col]):
                hot_row = start_index + np.argmax(bit_strings[start_index:, hot_col])
                failed = False

                break
        if failed:
            break

        if hot_col != start_index:
            bit_strings[:, [start_index, hot_col]] = bit_strings[:, [hot_col, start_index]]
            col_order[[start_index, hot_col]] = col_order[[hot_col, start_index]]

        if hot_row != start_index:
            bit_strings[[start_index, hot_row], :] = bit_strings[[hot_row, start_index], :]
            row_op[[start_index, hot_row], :] = row_op[[hot_row, start_index], :]

        cond_rows = np.logical_and(bit_strings[:, start_index], (row_range != start_index))

        bit_strings[cond_rows, :] = np.logical_xor(bit_strings[cond_rows, :], bit_strings[start_index, :][None, :])
        row_op[cond_rows, :] = np.logical_xor(row_op[cond_rows, :], row_op[start_index, :][None, :])

        start_index += 1

    return bit_strings, row_op, col_order, start_index
