from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from pauliarray.binary import bit_operations as bitops

INT2ZXBITS = [(0, 0), (1, 0), (0, 1), (1, 1)]
from scipy.optimize import linear_sum_assignment

# Basic operations


def merge_zx_strings(z_strings: "np.ndarray[np.bool]", x_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Merges z and x strings to create one zx string.

    Args:
        z_strings ("np.ndarray[np.bool]"): First string to concatenate (z)
        x_strings ("np.ndarray[np.bool]"): Second string to concatenate (x)

    Returns:
        "np.ndarray[np.bool]": Concatenated zx string.
    """

    return np.concatenate((z_strings, x_strings), axis=-1)


def split_zx_strings(zx_strings: "np.ndarray[np.bool]") -> tuple["np.ndarray[np.bool]", "np.ndarray[np.bool]"]:
    """
    Split one concatenated zx string into a z string and an x string.

    Args:
        zx_strings ("np.ndarray[np.bool]"): Concatenated zx string

    Returns:
        tuple["np.ndarray[np.bool]", "np.ndarray[np.bool]"]: Split z string and x string.
    """
    chain_len = zx_strings.shape[-1]
    assert chain_len % 2 == 0
    num_qubits = chain_len // 2

    z_strings = zx_strings[..., :num_qubits]
    x_strings = zx_strings[..., num_qubits:]

    return z_strings, x_strings


def zx_strings_to_active_bits(zx_strings: "np.ndarray[np.bool]"):

    z_strings, x_strings = split_zx_strings(zx_strings)

    return np.logical_or(z_strings, x_strings)


def flip_zx_strings(zx_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Flips an input zx string into an xz string.

    Args:
        zx_strings ("np.ndarray[np.bool]"): Input zx string to flip

    Returns:
        "np.ndarray[np.bool]": Flipped zx string (xz string)
    """

    assert zx_strings.shape[-1] % 2 == 0

    z_strings, x_strings = split_zx_strings(zx_strings)

    return merge_zx_strings(z_strings=x_strings, x_strings=z_strings)


def zx_strings_to_zips(zx_strings):

    num_qubits = zx_strings.shape[-1] // 2

    order = (np.arange(2 * num_qubits).reshape((2, num_qubits)).T).reshape((2 * num_qubits,))

    return zx_strings[..., order]


def zips_to_zx_strings(zips):

    num_qubits = zips.shape[-1] // 2

    order = (np.arange(2 * num_qubits).reshape((num_qubits, 2)).T).reshape((2 * num_qubits,))

    return zips[..., order]


def dot(
    zx_strings_1: "np.ndarray[np.bool]",
    zx_strings_2: "np.ndarray[np.bool]",
) -> "np.ndarray[np.int]":
    """
    Computes the binary dot product between two zx boolean strings.

    Args:
        zx_strings_1 ("np.ndarray[np.bool]"): First input zx string to compute the dot product with.
        zx_strings_2 ("np.ndarray[np.bool]"): Second input zx string to compute the dot product with.

    Returns:
        "np.ndarray[np.int]": Dot product of the two input arrays.
    """
    return bitops.dot(zx_strings_1, flip_zx_strings(zx_strings_2))


# Subspaces (Isotropic, Coisotropic, Lagrandian)


def is_orthogonal(zx_strings_1: "np.ndarray[np.bool]", zx_strings_2: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Check if the zx_strings of one string array are orthogonal to the zx_strings of an other string array element wise
    and using the simplectic dot product. Two zx_strings are orthogonal if the symplectic dot product is 0.

    Args:
        zx_strings_1 ("np.ndarray[np.bool]"): A list of zx_strings defining a subspace.
        Last dimension is along the length of the strings.
        zx_strings_2 ("np.ndarray[np.bool]"): A list of zx_strings defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        "np.ndarray[np.bool]": Wether or not the two input string are orthogonal to each other.
    """
    assert zx_strings_1.shape[-1] == zx_strings_2.shape[-1]

    return ~np.mod(dot(zx_strings_1, zx_strings_2), 2).astype(bool)


def is_isotropic(zx_strings: "np.ndarray[np.bool]") -> bool:
    """
    Checks if a given list of zx_strings defines an isotropic subspace.

    Args:
        zx_strings ("np.ndarray[np.bool]"): A list of zx_strings (2d-array of bools) defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        bool: Wether or not input is isotropic
    """
    assert zx_strings.ndim == 2

    return (
        zx_strings.shape[0] <= zx_strings.shape[1]
        and bitops.rank(zx_strings) == zx_strings.shape[0]
        and bool(np.all(is_orthogonal(zx_strings[:, None, :], zx_strings[None, :, :])))
    )


def is_coisotropic(zx_strings: "np.ndarray[np.bool]") -> bool:
    """
    Checks if a given list of zx_strings defines an coisotropic subspace.

    Args:
        zx_strings ("np.ndarray[np.bool]"): A list of zx_strings (2d-array of bools) defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        bool: Wether or not input is coisotropic
    """
    assert zx_strings.ndim == 2

    return zx_strings.shape[0] >= zx_strings.shape[1] and bitops.rank(zx_strings) == zx_strings.shape[0]


def is_lagrangian(zx_strings: "np.ndarray[np.bool]") -> bool:
    """
    Checks if a given list of zx_strings defines a Lagrangian subspace.

    Args:
        zx_strings ("np.ndarray[np.bool]"): A list of zx_strings (2d-array of bools) defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        bool: Wether or not input is lagrangian
    """
    assert is_isotropic(zx_strings)

    return 2 * zx_strings.shape[0] == zx_strings.shape[1]


def orthogonal_complement(zx_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Finds a list of zx_strings which or orthogonal in the symplectic sense to the provided zx_strings.
    For zx_strings of length 2*n, the number of returned zx_strings is 2*n - rank(zx_strings).
    The returned zx_strings might not be mutually orthogonal in the symplectic sense.

    Args:
        zx_strings ("np.ndarray[np.bool]"): A list of zx_strings (2d-array of bools) defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        "np.ndarray[np.bool]": A list of zx_strings defining the orthogonal completement.
    """
    return flip_zx_strings(bitops.orthogonal_complement(zx_strings))


def isotropic_subspace(orthogonal_zx_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Given a list of zx_strings which are mutually orthogonal in the smplectic sense, returns a list of zx_strings
    which are mutually orthogonal and linearly independent.

    Args:
        orthogonal_zx_strings ("np.ndarray[np.bool]"): A list of zx_strings (2d-array of bools) which are mutually
        orthogonal. Last dimension is along the length of the strings.

    Returns:
        "np.ndarray[np.bool]": A list of zx_strings defining the isotropic subspace.
    """
    assert orthogonal_zx_strings.ndim == 2
    assert np.all(is_orthogonal(orthogonal_zx_strings[:, None], orthogonal_zx_strings[None, :]))

    return zips_to_zx_strings(bitops.orthogonal_basis(zx_strings_to_zips(orthogonal_zx_strings)))


def coisotropic_subspace(orthogonal_zx_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Given a list of zx_strings which are mutually orthogonal in the smplectic sense, returns a list of zx_strings
    which are orthogonal to the provided zx_strings. For zx_strings of length 2*n, the number of returned zx_strings
    is 2*n - rank(zx_strings). The returned zx_strings might not be mutually orthogonal in the symplectic sense.

    Args:
        orthogonal_zx_strings ("np.ndarray[np.bool]"): A list of zx_strings (2d-array of bools) which are mutually
        orthogonal. Last dimension is along the length of the strings.

    Returns:
        "np.ndarray[np.bool]":A list of zx_strings defining the coisotropic subspace.
    """
    return orthogonal_complement(isotropic_subspace(orthogonal_zx_strings))


def lagrangian_subspace(orthogonal_zx_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Given a list of zx_strings of length 2*n which are mutually orthogonal in the smplectic sense, returns a list of
    n zx_strings which are mutually orthogonal and orthogonal to the provided zx_strings.

    Args:
        orthogonal_zx_strings ("np.ndarray[np.bool]"): A list of zx_strings (2d-array of bools) which are mutually
        orthogonal. Last dimension is along the length of the strings.

    Returns:
        "np.ndarray[np.bool]": A list of zx_strings defining the Lagrangian subspace.
    """

    tmp_zx_1 = gram_schmidt_orthogonalization(coisotropic_subspace(orthogonal_zx_strings))
    tmp_zx_2 = bitops.orthogonal_basis(zx_strings_to_zips(tmp_zx_1))

    return zips_to_zx_strings(tmp_zx_2)


def lagrangian_colagrangian_subspaces_old(lag_zx_strings: NDArray[np.bool_]):

    assert is_lagrangian(lag_zx_strings)

    num_qubits = lag_zx_strings.shape[0]
    the_range = np.arange(num_qubits)

    colag_zx_strings = np.zeros(lag_zx_strings.shape, dtype=bool)

    for i in range(num_qubits):
        ortho_lag_zx_strings = orthogonal_complement(lag_zx_strings[the_range != i, :])
        ortho_colag_zx_strings = orthogonal_complement(colag_zx_strings)
        ortho_inter_zx_strings = bitops.intersection(ortho_lag_zx_strings, ortho_colag_zx_strings)
        possible_conjugates = ortho_inter_zx_strings[~is_orthogonal(ortho_inter_zx_strings, lag_zx_strings[i, :])]

        colag_zx_strings[i, :] = possible_conjugates[np.argmin(bitops.bit_sum(possible_conjugates))]
        projections = np.mod(dot(lag_zx_strings[i + 1 :, None, :], colag_zx_strings[None, i : i + 1, :]), 2)

        lag_zx_strings[i + 1 :, :] = np.logical_xor(
            lag_zx_strings[i + 1 :, :], projections * lag_zx_strings[i : i + 1, :]
        )

    return lag_zx_strings, colag_zx_strings


def lagrangian_bitwise_colagrangian_subspaces(lag_zx_strings: NDArray[np.bool_]):
    """


    Based on : [1] T.-C. Yen, V. Verteletskyi, and A. F. Izmaylov, “Measuring All Compatible Operators in One Series of Single-Qubit Measurements Using Unitary Transformations,” J. Chem. Theory Comput., vol. 16, no. 4, pp. 2400–2409, Apr. 2020, doi: 10.1021/acs.jctc.0c00008.


    Args:
        lag_zx_strings (NDArray[np.bool_]): _description_

    Returns:
        _type_: _description_
    """

    assert is_lagrangian(lag_zx_strings)

    num_qubits = lag_zx_strings.shape[0]
    the_range = np.arange(num_qubits)
    done_mask = np.zeros((num_qubits,), dtype=bool)

    colag_zx_strings = np.zeros(lag_zx_strings.shape, dtype=bool)

    for i in range(num_qubits):

        lag_zx_bits = lag_zx_strings[:, [i, num_qubits + i]]
        j = np.argmax(np.any(lag_zx_bits, axis=1) * ~done_mask)

        if lag_zx_bits[j, 1]:  # anti commute with z
            colag_zx_strings[j, i] = True
        else:  # anti commute with x
            colag_zx_strings[j, num_qubits + i] = True

        projections = np.mod(dot(lag_zx_strings[the_range != j, None, :], colag_zx_strings[None, j : j + 1, :]), 2)

        lag_zx_strings[the_range != j, :] = np.logical_xor(
            lag_zx_strings[the_range != j, :], projections * lag_zx_strings[j : j + 1, :]
        )

        done_mask[j] = True

    return lag_zx_strings, colag_zx_strings


def conjugate_subspace(isotropic_zx_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Conctruct a conjugate subspace to the provided isotropic subspace. Each zx_string the conjugate subspace is
    orthogonal ins the symplectic sense to all provided zx_strings of the isotropic subspace,
    except for one : its conjugate.

    Args:
        isotropic_zx_strings ("np.ndarray[np.bool]"): Isotropic subspace.

    Returns:
        "np.ndarray[np.bool]": Conjugate subspace to the input isotropic subspace.
    """

    assert is_isotropic(isotropic_zx_strings)

    conj_subspace = np.zeros(isotropic_zx_strings.shape, dtype=bool)

    for i in range(isotropic_zx_strings.shape[0]):
        # find a subspace orthogonal to original subspace with element i
        ortho_subspace_minus_i = orthogonal_complement(
            isotropic_zx_strings[np.arange(isotropic_zx_strings.shape[0]) != i, :]
        )

        assert np.all(
            is_orthogonal(
                isotropic_zx_strings[np.arange(isotropic_zx_strings.shape[0]) != i, None],
                ortho_subspace_minus_i[None, :],
            )
        )

        # subspace orthogonal to all elements in the conjugate subspace up to this point
        ortho_conj_subspace = orthogonal_complement(conj_subspace)

        assert np.all(is_orthogonal(conj_subspace[:, None], ortho_conj_subspace[None, :]))

        # The intersection of these two spaces
        ortho_intersection = bitops.intersection(ortho_subspace_minus_i, ortho_conj_subspace)

        # Select the element in the intersection which are conjugate the the element from the Lagrangian
        possible_conjugates = ortho_intersection[~is_orthogonal(ortho_intersection, isotropic_zx_strings[i, :])]

        if possible_conjugates.shape[0] == 0:
            raise RuntimeError()

        conj_subspace[i, :] = possible_conjugates[np.argmin(bitops.bit_sum(possible_conjugates))]

    return conj_subspace


def row_ech_lagrangian_colagrangian(
    lag_zx_strings: NDArray[np.bool_], colag_zx_strings: NDArray[np.bool_]
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Finds and apply a transformation on a pair of Lagragian and co-Lagrangian subspaces to simplify the co-Lagrangian subspace.

    Args:
        lag_zx_strings (NDArray[np.bool_]): _description_
        colag_zx_strings (NDArray[np.bool_]): _description_

    Returns:
        Tuple[NDArray[np.bool_], NDArray[np.bool_]]: _description_
    """

    num_qubits = lag_zx_strings.shape[0]

    tmp_strings = np.concatenate((colag_zx_strings, np.eye(num_qubits, dtype=bool)), axis=-1)

    row_tmp_strings = bitops.row_echelon(tmp_strings)

    transformation = row_tmp_strings[:, -num_qubits:]

    return transform_lagrangian_colagrangian(lag_zx_strings, colag_zx_strings, transformation)


def row_ech_qubit_lagrangian_colagrangian(
    lag_zx_strings: NDArray[np.bool_], colag_zx_strings: NDArray[np.bool_]
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Finds and apply a transformation on a pair of Lagragian and co-Lagrangian subspaces to simplify the co-Lagrangian subspace.

    Args:
        lag_zx_strings (NDArray[np.bool_]): _description_
        colag_zx_strings (NDArray[np.bool_]): _description_

    Returns:
        Tuple[NDArray[np.bool_], NDArray[np.bool_]]: _description_
    """

    num_qubits = lag_zx_strings.shape[0]

    order = (np.arange(2 * num_qubits).reshape((2, num_qubits)).T).reshape((2 * num_qubits,))
    tmp_strings = np.concatenate((colag_zx_strings[:, order], np.eye(num_qubits, dtype=bool)), axis=-1)

    row_tmp_strings = bitops.row_echelon(tmp_strings)

    transformation = row_tmp_strings[:, -num_qubits:]

    return transform_lagrangian_colagrangian(lag_zx_strings, colag_zx_strings, transformation)


def transform_lagrangian_colagrangian(
    lag_zx_strings: NDArray[np.bool_], colag_zx_strings: NDArray[np.bool_], lag_transformation: NDArray[np.bool_]
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Finds and apply a transformation on a pair of Lagragian and co-Lagrangian subspaces to simplify the co-Lagrangian subspace.

    Args:
        lag_zx_strings (NDArray[np.bool_]): _description_
        colag_zx_strings (NDArray[np.bool_]): _description_

    Returns:
        Tuple[NDArray[np.bool_], NDArray[np.bool_]]: _description_
    """

    colag_transformation = (bitops.inv(lag_transformation)).T

    new_lag_zx_strings = np.mod(colag_transformation.astype(np.uint8) @ lag_zx_strings.astype(np.uint8), 2).astype(
        np.bool_
    )
    new_colag_zx_strings = np.mod(lag_transformation.astype(np.uint8) @ colag_zx_strings.astype(np.uint8), 2).astype(
        np.bool_
    )

    return new_lag_zx_strings, new_colag_zx_strings


def gram_schmidt_orthogonalization(zx_strings: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
    """
    Performs a Gram Schmidt orthogonalization procedure in the symplectic space.

    Args:
        zx_strings ("np.ndarray[np.bool]"): Input space.

    Returns:
        "np.ndarray[np.bool]": Bit strings which are mutually symplectic orthogonal spawning the same space as the input.
    """
    assert zx_strings.ndim == 2

    num_strings, num_qubits = zx_strings.shape

    working_zx_strings = zx_strings.copy()

    for _ in range(num_strings):
        colinear = np.mod(dot(working_zx_strings[:, None, :], working_zx_strings[None, :, :]), 2)
        colinear_pairs = np.where(colinear)

        if len(colinear_pairs[0]) > 0:
            colinear_pair = [colinear_pairs[0][0], colinear_pairs[1][0]]

            pair_mask = np.zeros((working_zx_strings.shape[0],), dtype=bool)
            pair_mask[colinear_pair] = True
            factors = colinear[~pair_mask, :][:, np.flip(colinear_pair)]

            new_zx_strings = np.zeros((working_zx_strings.shape[0] - 1, num_qubits), dtype=bool)

            new_zx_strings[0, :] = working_zx_strings[colinear_pair[0]]

            new_zx_strings[1:, :] = np.logical_xor(
                working_zx_strings[~pair_mask, :], bitops.matmul(factors, working_zx_strings[pair_mask, :])
            )

            working_zx_strings = new_zx_strings
        else:
            break

    return working_zx_strings


def diagonal_assignement(zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Reorder zx_strings so that the nth one is active on the nth qubit.

    Args:
        zx_strings (z2c.Z2ChainArray): _description_

    Returns:
        z2c.Z2ChainArray: _description_
    """

    assert 2 * zx_strings.shape[0] == zx_strings.shape[1]

    active_bits = zx_strings_to_active_bits(zx_strings)

    row_ind, col_ind = linear_sum_assignment(active_bits.T.astype(int), maximize=True)

    new_zx_strings = zx_strings[col_ind]

    assert np.all(row_ind == np.arange(active_bits.shape[1]))
    assert is_diagonal_assigned(new_zx_strings)

    return new_zx_strings


def is_diagonal_assigned(zx_strings: NDArray[np.bool_]) -> bool:

    z_strings, x_strings = split_zx_strings(zx_strings)

    active_bits = np.logical_or(z_strings, x_strings)

    return ~np.any(active_bits.diagonal() == 0)
