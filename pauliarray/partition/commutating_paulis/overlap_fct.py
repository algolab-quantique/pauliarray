from typing import Callable, List, Protocol, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops


class HasPaulis(Protocol):
    paulis: pa.PauliArray


def _commutation_adjacency_to_overlap_parts_idx_networkx(commutation_adjacency) -> List[List[int]]:
    """
    Uses Networkx greedy_color to

    Args:
        commutation_adjacency (_type_): _description_
        strategy (str, optional): _description_. Defaults to "largest_first".

    Returns:
        List[List[int]]: _description_
    """

    graph = nx.from_numpy_array(commutation_adjacency)

    cliques = list(nx.find_cliques(graph))

    return cliques


def partition_general_commutating(
    pauli_obj: HasPaulis,
    commutation_adjacency_to_parts_idx: Callable = _commutation_adjacency_to_overlap_parts_idx_networkx,
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

    print(commutation_adjacency.astype(int))

    return commutation_adjacency_to_parts_idx(commutation_adjacency)
