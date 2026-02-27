import numpy as np
from numpy.typing import NDArray


def build_identity_matrix(size: int) -> NDArray[np.bool_]:

    return np.eye(size, dtype=np.bool_)


def build_parity_matrix(size: int) -> NDArray[np.bool_]:

    return np.tri(size, dtype=np.bool_)


def build_bravyi_kitaev_matrix(size: int) -> NDArray[np.bool_]:

    mapping_matrix = np.eye(size, dtype=np.bool_)

    for i in range(1, size + 1, 2):
        if np.log2(i + 1) % 1 == 0:
            mapping_matrix[i, : i + 1] = True
        else:
            mapping_matrix[i, 2 ** int(np.log2(i + 1)) : i + 1] = True

    return mapping_matrix


def build_heavyside_matrix(size: int) -> NDArray[np.bool_]:

    return np.tri(size, k=-1, dtype=np.bool_)


def build_random_invertible_matrix(size: int) -> NDArray[np.bool_]:

    random_mat = np.tril(np.random.choice([0, 1], size=(size, size)).astype(bool))

    invertible_matrix = np.logical_or(random_mat, np.identity(size, dtype=bool))

    random_order = np.arange(size)
    np.random.shuffle(random_order)
    invertible_matrix = invertible_matrix[random_order, :][:, random_order]

    return invertible_matrix
