from typing import Callable, List, Protocol, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops


class HasPaulis(Protocol):
    paulis: pa.PauliArray


def _commutation_adjacency_to_exclusive_parts_idx_networkx(
    commutation_adjacency, strategy="largest_first"
) -> List[List[int]]:
    """
    Uses Networkx greedy_color to

    Args:
        commutation_adjacency (_type_): _description_
        strategy (str, optional): _description_. Defaults to "largest_first".

    Returns:
        List[List[int]]: _description_
    """

    graph = nx.from_numpy_array(~commutation_adjacency)

    coloring_dict = nx.greedy_color(graph, strategy=strategy)

    num_colors = np.max(list(coloring_dict.values())) + 1
    idx_in_parts = [[] for _ in range(num_colors)]
    for idx, color in coloring_dict.items():
        idx_in_parts[color].append(idx)

    return idx_in_parts


def partition_bitwise_commutating(
    pauli_obj: HasPaulis,
    commutation_adjacency_to_parts_idx: Callable = _commutation_adjacency_to_exclusive_parts_idx_networkx,
) -> List[NDArray[np.int_]]:
    """
    Partition a PauliArray based on bitwise commutation.

    Args:
        paulis (pa.PauliArray): _description_
        commutation_adjacency_to_parts_idx (Callable): A function which takes a commutation adjacency matrix and returns a list of parts given as linear indices

    Returns:
        List[NDArray[np.int_]]: Parts given as linear indices
    """

    paulis = pauli_obj.paulis.flatten()

    commutation_adjacency = paulis[:, None].bitwise_commute_with(paulis[None, :])

    return commutation_adjacency_to_parts_idx(commutation_adjacency)


def partition_general_commutating(
    pauli_obj: HasPaulis,
    commutation_adjacency_to_parts_idx: Callable = _commutation_adjacency_to_exclusive_parts_idx_networkx,
) -> List[NDArray[np.int_]]:
    """
    Partition a PauliArray based on general commutation.

    Args:
        paulis (PauliArray): _description_
        commutation_adjacency_to_parts_idx (Callable): A function which takes a commutation adjacency matrix and returns a list of parts given as linear indices

    Returns:
        List[NDArray[np.int_]]: Parts given as linear indices
    """
    paulis = pauli_obj.paulis.flatten()

    commutation_adjacency = paulis[:, None].commute_with(paulis[None, :])

    return commutation_adjacency_to_parts_idx(commutation_adjacency)


def partition_same_x(pauli_obj: HasPaulis) -> List[NDArray[np.int_]]:
    """
    Builds a ExclusiveArrayPartition for a PauliArray based on the same X approach.

    Args:
        paulis (pa.PauliArray): _description_

    Returns:
        List[NDArray[np.int_]]: Parts given as linear indices
    """
    paulis = pauli_obj.paulis.flatten()

    y_mask = np.logical_and(paulis.x_strings, paulis.z_strings)
    y_parity = np.mod(np.sum(y_mask, axis=-1), 2)

    x_and_y_parity_chains = np.concatenate((paulis.x_strings, y_parity[..., None]), axis=-1)

    _, flat_inverse = bitops.fast_flat_unique_bit_string(x_and_y_parity_chains, return_inverse=True)

    parts_idx = _part_idx_from_unique_inverse(flat_inverse)

    return parts_idx


def partition_same_x_plus_special(
    pauli_obj: HasPaulis,
    commutation_adjacency_to_parts_idx: Callable = _commutation_adjacency_to_exclusive_parts_idx_networkx,
) -> List[NDArray[np.int_]]:
    """
    Builds a ExclusiveArrayPartition for a PauliArray based on the same X approach. Then assemble commutating same x partitions into larger partition.

    Args:
        paulis (pa.PauliArray): _description_

    Returns:
        List[NDArray[np.int_]]: Parts given as linear indices
    """

    paulis = pauli_obj.paulis.flatten()

    parts_idx_same_x = partition_same_x(paulis)

    number_of_same_x_parts = len(parts_idx_same_x)

    first_same_x_idx = [idx_in_part[0] for idx_in_part in parts_idx_same_x]
    unique_x_strings = paulis[first_same_x_idx].x_strings
    unique_x_paulis = pa.PauliArray(np.zeros(unique_x_strings.shape, dtype=bool), unique_x_strings)

    do_commute_with_x = paulis[:, None].commute_with(unique_x_paulis[None, :])

    comm_matrix = np.zeros((number_of_same_x_parts, number_of_same_x_parts), dtype=bool)
    anti_matrix = np.zeros((number_of_same_x_parts, number_of_same_x_parts), dtype=bool)
    for i, idx in enumerate(parts_idx_same_x):
        comm_matrix[i, :] = np.all(do_commute_with_x[idx, :], axis=0)
        anti_matrix[i, :] = np.all(~do_commute_with_x[idx, :], axis=0)

    groups_commutation_adjacency = np.logical_or(
        np.logical_and(comm_matrix, comm_matrix.T), np.logical_and(anti_matrix, anti_matrix.T)
    )

    commuting_groups_idx = commutation_adjacency_to_parts_idx(groups_commutation_adjacency)

    parts_idx = list()
    for commuting_group_idx in commuting_groups_idx:
        part_idx = list()
        for i in commuting_group_idx:
            part_idx += parts_idx_same_x[i]
        parts_idx.append(part_idx)

    return parts_idx


def _part_idx_from_unique_inverse(inverse) -> List[List[int]]:
    number_of_parts = max(inverse) + 1
    all_indices = np.arange(len(inverse))
    parts_idx = list()
    for i_part in range(number_of_parts):
        parts_idx.append(all_indices[inverse == i_part].tolist())

    return parts_idx
