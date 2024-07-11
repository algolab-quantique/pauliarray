from typing import Callable, List, Protocol, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops


class PauliObj(Protocol):
    paulis: pa.PauliArray


def _commutation_adjacency_to_parts_idx_networkx(commutation_adjacency, strategy="largest_first") -> List[List[int]]:
    """
    Uses Networkx greedy_color to

    Args:
        commutation_adjacency (_type_): _description_
        strategy (str, optional): _description_. Defaults to "largest_first".

    Returns:
        List[List[int]]: _description_
    """
    edges = list(zip(*np.where(np.triu(~commutation_adjacency, k=1))))

    graph = nx.Graph()
    graph.add_nodes_from(range(commutation_adjacency.shape[0]))
    graph.add_edges_from(edges)

    coloring_dict = nx.greedy_color(graph, strategy=strategy)

    num_colors = np.max(list(coloring_dict.values())) + 1
    idx_in_parts = [[] for _ in range(num_colors)]
    for idx, color in coloring_dict.items():
        idx_in_parts[color].append(idx)

    return idx_in_parts


def partition_bitwise_commutating(
    pauli_obj: PauliObj, commutation_adjacency_to_parts_idx: Callable = _commutation_adjacency_to_parts_idx_networkx
) -> List[NDArray[np.int_]]:
    """
    Builds a ExclusiveArrayPartition for a PauliArray based on bitwise commutation.

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
    pauli_obj: PauliObj, commutation_adjacency_to_parts_idx: Callable = _commutation_adjacency_to_parts_idx_networkx
) -> List[NDArray[np.int_]]:
    """
    Builds a ExclusiveArrayPartition for a PauliArray based on general commutation.

    Args:
        paulis (PauliArray): _description_
        commutation_adjacency_to_parts_idx (Callable): A function which takes a commutation adjacency matrix and returns a list of parts given as linear indices

    Returns:
        List[NDArray[np.int_]]: Parts given as linear indices
    """
    paulis = pauli_obj.paulis.flatten()

    commutation_adjacency = paulis[:, None].commute_with(paulis[None, :])

    return commutation_adjacency_to_parts_idx(commutation_adjacency)
