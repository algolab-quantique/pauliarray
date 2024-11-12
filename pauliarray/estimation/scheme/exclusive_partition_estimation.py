from typing import Any, Callable, List, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

import pauliarray.pauli.operator_array_type_1 as opa
import pauliarray.pauli.pauli_array as pa
from pauliarray.estimation.base_estimators import BaseEstimator, DiagonalEstimator, GeneralEstimator
from pauliarray.pauli.pauli_array import PauliArray
from pauliarray.state.nqubit_state import NQubitState


class EstimatePauliObject(Protocol):

    paulis: pa.PauliArray

    def with_new_paulis(self, new_paulis: pa.PauliArray) -> "EstimatePauliObject": ...

    def expectation_values_from_paulis(self, paulis_expectation_values: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def covariances_from_paulis(self, paulis_covariances: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def partition(self, parts_flat_idx: List[NDArray[np.int64]]) -> List["EstimatePauliObject"]: ...
    def partition_with_fct(self, partition_fct: Callable) -> List["EstimatePauliObject"]: ...


class ExclusivePartitionEstimationScheme(object):

    def __init__(
        self,
        pauli_obj: EstimatePauliObject,
        ll_estimator: BaseEstimator,
        partition_fct: Callable,
        diagonalisation_fct: Union[None, Callable],
    ):

        self._pauli_obj = pauli_obj
        self._ll_estimator = ll_estimator
        self._partition_fct = partition_fct
        self._diagonalisation_fct = diagonalisation_fct

    def partition(self, pauli_obj: EstimatePauliObject):

        parts_flat_idx = self._partition_fct(pauli_obj)
        parts = pauli_obj.partition(parts_flat_idx)

        return parts_flat_idx, parts

    def diagonalise_parts(self, parts: EstimatePauliObject):

        diag_parts = []
        parts_factors = []
        parts_transformation = []
        for part in parts:
            diag_paulis, factors, transformation = self._diagonalisation_fct(part.paulis)
            diag_parts.append(part.with_new_paulis(diag_paulis))
            parts_factors.append(factors)
            parts_transformation.append(transformation)

        return diag_parts, parts_factors, parts_transformation

    def assemble_paulis_expectation_values(self, parts_flat_idx, parts_expectation_values):

        array = np.zeros(self._pauli_obj.paulis.size, dtype=type(parts_expectation_values[0][0]))
        for part_expectation_values, part_flat_idx in zip(parts_expectation_values, parts_flat_idx):
            array[part_flat_idx] = part_expectation_values

        return array

    def estimate_on_state(self, state: Any):
        """
        Estimate the expectation value of the pauli object.

        Args:
            state_circuit (QuantumCircuit): A state given in a form compatible with the scheme

        Returns:
            NDArray: _description_
        """

        paulis = self._pauli_obj.paulis

        parts_flat_idx, parts = self.partition(paulis)

        if isinstance(self._ll_estimator, DiagonalEstimator) and isinstance(self._diagonalisation_fct, Callable):
            diag_parts, parts_factors, parts_transformation = self.diagonalise_parts(parts)

            if isinstance(parts_transformation[0], opa.OperatorArrayType1):

                nqubit_state: NQubitState = state

                transformed_states = []
                for part_transformation in parts_transformation:
                    transformed_nqubit_state = nqubit_state.apply_operator_array(part_transformation)
                    transformed_states.append(transformed_nqubit_state)

            if isinstance(parts_transformation[0], QuantumCircuit):

                circuit_state: QuantumCircuit = state

                transformed_states = []
                for part_transformation in parts_transformation:
                    transformed_circuit_state = circuit_state.compose(part_transformation)
                    transformed_states.append(transformed_circuit_state)

            parts_expectation_values = []
            for diag_part, part_factors, transformed_state in zip(diag_parts, parts_factors, transformed_states):

                pre_part_expectation_values = self._ll_estimator.estimate_paulis_on_state(diag_part, transformed_state)
                part_expectation_values = part_factors * pre_part_expectation_values
                parts_expectation_values.append(part_expectation_values)

        paulis_expectation_values = self.assemble_paulis_expectation_values(parts_flat_idx, parts_expectation_values)

        pauli_obj_expectation_value = self._pauli_obj.expectation_values_from_paulis(paulis_expectation_values)
        # pauli_obj_covariance = self._pauli_obj.covariances_from_paulis(paulis_expectation_values)

        return pauli_obj_expectation_value
