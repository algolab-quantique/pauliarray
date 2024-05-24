from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from pauliarray.binary import bit_operations as bitops

INT2ZXBITS = [(0, 0), (1, 0), (0, 1), (1, 1)]


# Basic operations


def merge_zx_strings(z_strings: NDArray[np.bool_], x_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Merges z and x strings to create one zx string.

    Args:
        z_strings (NDArray[np.bool_]): First string to concatenate (z)
        x_strings (NDArray[np.bool_]): Second string to concatenate (x)

    Returns:
        NDArray[np.bool_]: Concatenated zx string.
    """

    return np.concatenate((z_strings, x_strings), axis=-1)


def split_zx_strings(zx_strings: NDArray[np.bool_]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Split one concatenated zx string into a z string and an x string.

    Args:
        zx_strings (NDArray[np.bool_]): Concatenated zx string

    Returns:
        tuple[NDArray[np.bool_], NDArray[np.bool_]]: Split z string and x string.
    """
    chain_len = zx_strings.shape[-1]
    assert chain_len % 2 == 0
    num_qubits = chain_len // 2

    z_strings = zx_strings[..., :num_qubits]
    x_strings = zx_strings[..., num_qubits:]

    return z_strings, x_strings


def flip_zx(zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Flips an input zx string into an xz string.

    Args:
        zx_strings (NDArray[np.bool_]): Input zx string to flip

    Returns:
        NDArray[np.bool_]: Flipped zx string (xz string)
    """

    assert zx_strings.shape[-1] % 2 == 0

    z_strings, x_strings = split_zx_strings(zx_strings)

    return merge_zx_strings(z_strings=x_strings, x_strings=z_strings)


def dot(
    zx_strings_1: NDArray[np.bool_],
    zx_strings_2: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """
    Computes the binary dot product between two zx boolean strings.

    Args:
        zx_strings_1 (NDArray[np.bool_]): First input zx string to compute the dot product with.
        zx_strings_2 (NDArray[np.bool_]): Second input zx string to compute the dot product with.

    Returns:
        NDArray[np.int_]: Dot product of the two input arrays.
    """
    return bitops.dot(zx_strings_1, flip_zx(zx_strings_2))


# Subspaces (Isotropic, Coisotropic, Lagrandian)


def is_orthogonal(zx_strings_1: NDArray[np.bool_], zx_strings_2: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Check if the zx_strings of one string array are orthogonal to the zx_strings of an other string array element wise
    and using the simplectic dot product. Two zx_strings are orthogonal if the symplectic dot product is 0.

    Args:
        zx_strings_1 (NDArray[np.bool_]): A list of zx_strings defining a subspace.
        Last dimension is along the length of the strings.
        zx_strings_2 (NDArray[np.bool_]): A list of zx_strings defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        NDArray[np.bool_]: Wether or not the two input string are orthogonal to each other.
    """
    assert zx_strings_1.shape[-1] == zx_strings_2.shape[-1]

    return ~np.mod(dot(zx_strings_1, zx_strings_2), 2).astype(bool)


def is_isotropic(zx_strings: NDArray[np.bool_]) -> bool:
    """
    Checks if a given list of zx_strings defines an isotropic subspace.

    Args:
        zx_strings (NDArray[np.bool_]): A list of zx_strings (2d-array of bools) defining a subspace.

    Returns:
        bool: Wether or not input is isotropic
    """
    assert zx_strings.ndim == 2

    return (
        zx_strings.shape[0] <= zx_strings.shape[1]
        and bitops.rank(zx_strings) == zx_strings.shape[0]
        and bool(np.all(is_orthogonal(zx_strings[:, None, :], zx_strings[None, :, :])))
    )


def is_coisotropic(zx_strings: NDArray[np.bool_]) -> bool:
    """
    Checks if a given list of zx_strings defines an coisotropic subspace.

    Args:
        zx_strings (NDArray[np.bool_]): A list of zx_strings (2d-array of bools) defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        bool: Wether or not input is coisotropic
    """
    assert zx_strings.ndim == 2

    return zx_strings.shape[0] >= zx_strings.shape[1] and bitops.rank(zx_strings) == zx_strings.shape[0]


def is_lagrangian(zx_strings: NDArray[np.bool_]) -> bool:
    """
    Checks if a given list of zx_strings defines a Lagrangian subspace.

    Args:
        zx_strings (NDArray[np.bool_]): A list of zx_strings (2d-array of bools) defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        bool: Wether or not input is lagrangian
    """
    assert is_isotropic(zx_strings)

    return 2 * zx_strings.shape[0] == zx_strings.shape[1]


def orthogonal_complement(zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Finds a list of zx_strings which or orthogonal in the symplectic sense to the provided zx_strings.
    For zx_strings of length 2*n, the number of returned zx_strings is 2*n - rank(zx_strings).
    The returned zx_strings might not be mutually orthogonal in the symplectic sense.

    Args:
        zx_strings (NDArray[np.bool_]): A list of zx_strings (2d-array of bools) defining a subspace.
        Last dimension is along the length of the strings.

    Returns:
        NDArray[np.bool_]: A list of zx_strings defining the orthogonal completement.
    """
    return flip_zx(bitops.orthogonal_complement(zx_strings))


def isotropic_subspace(orthogonal_zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Given a list of zx_strings which are mutually orthogonal in the smplectic sense, returns a list of zx_strings
    which are mutually orthogonal and linearly independent.

    Args:
        orthogonal_zx_strings (NDArray[np.bool_]): A list of zx_strings (2d-array of bools) which are mutually
        orthogonal. Last dimension is along the length of the strings.

    Returns:
        NDArray[np.bool_]: A list of zx_strings defining the isotropic subspace.
    """
    assert orthogonal_zx_strings.ndim == 2
    assert np.all(is_orthogonal(orthogonal_zx_strings[:, None], orthogonal_zx_strings[None, :]))

    return bitops.orthogonal_basis(orthogonal_zx_strings)


def coisotropic_subspace(orthogonal_zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Given a list of zx_strings which are mutually orthogonal in the smplectic sense, returns a list of zx_strings
    which are orthogonal to the provided zx_strings. For zx_strings of length 2*n, the number of returned zx_strings
    is 2*n - rank(zx_strings). The returned zx_strings might not be mutually orthogonal in the symplectic sense.

    Args:
        orthogonal_zx_strings (NDArray[np.bool_]): A list of zx_strings (2d-array of bools) which are mutually
        orthogonal. Last dimension is along the length of the strings.

    Returns:
        NDArray[np.bool_]:A list of zx_strings defining the coisotropic subspace.
    """
    return orthogonal_complement(isotropic_subspace(orthogonal_zx_strings))


def lagrangian_subspace(orthogonal_zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Given a list of zx_strings of length 2*n which are mutually orthogonal in the smplectic sense, returns a list of
    n zx_strings which are mutually orthogonal and orthogonal to the provided zx_strings.

    Args:
        orthogonal_zx_strings (NDArray[np.bool_]): A list of zx_strings (2d-array of bools) which are mutually
        orthogonal. Last dimension is along the length of the strings.

    Returns:
        NDArray[np.bool_]: A list of zx_strings defining the Lagrangian subspace.
    """

    return bitops.orthogonal_basis(gram_schmidt_orthogonalization(coisotropic_subspace(orthogonal_zx_strings)))


def conjugate_subspace(isotropic_zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Conctruct a conjugate subspace to the provided isotropic subspace. Each zx_string the conjugate subspace is
    orthogonal ins the symplectic sense to all provided zx_strings of the isotropic subspace,
    except for one : its conjugate.

    Args:
        isotropic_zx_strings (NDArray[np.bool_]): Isotropic subspace.

    Returns:
        NDArray[np.bool_]: Conjugate subspace to the input isotropic subspace.
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


def gram_schmidt_orthogonalization(zx_strings: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Performs a Gram Schmidt orthogonalization procedure in the symplectic space.

    Args:
        zx_strings (NDArray[np.bool_]): Input space.

    Returns:
        NDArray[np.bool_]: Bit strings which are mutually symplectic orthogonal spawning the same space as the input.
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
