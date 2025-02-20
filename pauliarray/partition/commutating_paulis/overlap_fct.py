from typing import Callable, List, Protocol, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from pauliarray.binary import bit_operations as bitops
from pauliarray.utils.protocols import HasPaulis


def _commutation_adjacency_to_overlap_parts_idx_networkx(commutation_adjacency) -> List[List[int]]:
    """
    Uses Networkx find_cliques to identify indices of all commuting elements cliques.

    Args:
        commutation_adjacency (_type_): A adjacency matrix with element ij to True if element i and j commute.
        strategy (str, optional): Strategy to be passed to Networkx find_cliques. Defaults to "largest_first".

    Returns:
        List[List[int]]: List of all cliques given by indices
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

    return commutation_adjacency_to_parts_idx(commutation_adjacency)
