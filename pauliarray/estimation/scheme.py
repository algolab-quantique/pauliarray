from typing import Callable, List, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.pauli_array as pa


class HasPaulis(Protocol):
    paulis: pa.PauliArray


class EstimatePaulis(Protocol, HasPaulis):

    def expectation_values_from_paulis(self, paulis_expectation_values: NDArray[np.float_]) -> NDArray[np.float_]: ...
    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float_]) -> NDArray[np.float_]: ...
    def partition(self, parts_flat_idx: List[NDArray[np.int_]]) -> List[HasPaulis]: ...
    def partition_with_fct(self, partition_fct: Callable) -> List[HasPaulis]: ...


# def estimate_diagonal_paulis


class EstimationSchemaExclusivePartition(object):

    def __init__(self, pauli_obj: EstimatePaulis, partition_fct: Callable, diagonalisation_fct: Callable):

        self._pauli_obj = pauli_obj
        self._partition_fct = partition_fct
        self._diagonalisation_fct = diagonalisation_fct

        self._parts = None

    def run(self):

        self._parts = self._pauli_obj.partition_with_fct(self._partition_fct)
