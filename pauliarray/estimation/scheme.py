from typing import Callable, List, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.pauli_array as pa


class HasPaulis(Protocol):
    paulis: pa.PauliArray

    def with_new_paulis(self, new_paulis: pa.PauliArray) -> "HasPaulis": ...


class EstimatePaulis(Protocol):

    def expectation_values_from_paulis(self, paulis_expectation_values: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def partition(self, parts_flat_idx: List[NDArray[np.int64]]) -> List[HasPaulis]: ...
    def partition_with_fct(self, partition_fct: Callable) -> List[HasPaulis]: ...


# def estimate_diagonal_paulis


class EstimationSchemeExclusivePartition(object):

    def __init__(self, pauli_obj: EstimatePaulis, partition_fct: Callable, diagonalisation_fct: Callable):

        self._pauli_obj = pauli_obj
        self._partition_fct = partition_fct
        self._diagonalisation_fct = diagonalisation_fct

        self.parts_flat_idx, self.diag_parts, self.parts_factors, self.parts_transformation = self.prepare()

    def prepare(self):

        parts_flat_idx = self._partition_fct(self._pauli_obj)
        parts = self._pauli_obj.partition(parts_flat_idx)

        diag_parts = []
        parts_factors = []
        parts_transformation = []
        for part in parts:
            diag_paulis, factors, transformation = self._diagonalisation_fct(part.paulis)
            diag_parts.append(part.with_new_paulis(diag_paulis))
            parts_factors.append(factors)
            parts_transformation.append(transformation)

        return parts_flat_idx, diag_parts, parts_factors, parts_transformation
