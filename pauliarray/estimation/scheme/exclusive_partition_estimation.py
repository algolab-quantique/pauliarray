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
    def standard_deviations_from_paulis(self, flat_paulis_covariances: NDArray[np.float64], shots: int): ...
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

        expectation_values = np.zeros(self._pauli_obj.paulis.size, dtype=type(parts_expectation_values[0][0]))
        for part_expectation_values, part_flat_idx in zip(parts_expectation_values, parts_flat_idx):
            expectation_values[part_flat_idx] = part_expectation_values

        return expectation_values

    def assemble_paulis_covariances(self, parts_flat_idx, parts_covariances):

        covariances = np.zeros(
            (self._pauli_obj.paulis.size, self._pauli_obj.paulis.size), dtype=type(parts_covariances[0][0][0])
        )
        for part_covariances, part_flat_idx in zip(parts_covariances, parts_flat_idx):
            covariances[np.ix_(part_flat_idx, part_flat_idx)] = part_covariances

        return covariances

    def assemble_paulis_shots(self, parts_flat_idx, parts_shots):
        paulis_shots = np.zeros(self._pauli_obj.paulis.size)
        for part_shots, parts_flat_idx in zip(parts_shots, parts_flat_idx):
            paulis_shots[parts_flat_idx] += part_shots

        return paulis_shots

    def prepare_transformed_states(self, parts_transformation, state):

        if isinstance(parts_transformation[0], opa.OperatorArrayType1):

            nqubit_state: NQubitState = state

            transformed_states = []
            for part_transformation in parts_transformation:
                transformed_nqubit_state = nqubit_state.apply_operator_array(part_transformation)
                transformed_states.append(transformed_nqubit_state)

            return transformed_states

        if isinstance(parts_transformation[0], QuantumCircuit):

            circuit_state: QuantumCircuit = state

            transformed_states = []
            for part_transformation in parts_transformation:
                transformed_circuit_state = circuit_state.compose(part_transformation)
                transformed_states.append(transformed_circuit_state)

            return transformed_states

        return NotImplemented

    def prepare_batch(self, diag_parts, transformed_states, return_cov):

        batch_paulis = []
        batch_state = []
        for diag_paulis, transformed_state in zip(diag_parts, transformed_states):
            diag_paulis: pa.PauliArray = diag_paulis
            if return_cov:
                ij_diag_paulis, _ = diag_paulis[:, None].compose(diag_paulis[None, :])
                concat_paulis = pa.concatenate((diag_paulis, ij_diag_paulis.flatten()), axis=0)
                batch_paulis.append(concat_paulis)
            else:
                batch_paulis.append(diag_paulis)
            batch_state.append(transformed_state)

        return batch_paulis, batch_state

    def extract_parts_expectation_values(self, batch_expectation_values, parts_factors):

        parts_expectation_values = []
        for concat_paulis_expectation_values, part_factors in zip(batch_expectation_values, parts_factors):
            n_paulis = len(part_factors)
            diag_paulis_expectation_values = concat_paulis_expectation_values[:n_paulis]

            part_expectation_values = part_factors * diag_paulis_expectation_values
            parts_expectation_values.append(part_expectation_values)

        return parts_expectation_values

    def extract_parts_expectation_values_and_covariances(self, batch_expectation_values, parts_factors):

        parts_expectation_values = []
        parts_covariances = []
        for concat_paulis_expectation_values, part_factors in zip(batch_expectation_values, parts_factors):
            n_paulis = len(part_factors)
            diag_paulis_expectation_values = concat_paulis_expectation_values[:n_paulis]

            part_expectation_values = part_factors * diag_paulis_expectation_values
            parts_expectation_values.append(part_expectation_values)

            ij_diag_paulis_expectation_values = concat_paulis_expectation_values[n_paulis:].reshape(
                (n_paulis, n_paulis)
            )
            part_covariances = (
                part_factors[:, None]
                * part_factors[None, :]
                * (
                    ij_diag_paulis_expectation_values
                    - diag_paulis_expectation_values[:, None] * diag_paulis_expectation_values[None, :]
                )
            )

            parts_covariances.append(part_covariances)

        return parts_expectation_values, parts_covariances

    def estimate_on_state(self, state: Any, return_cov=False, return_std=False):
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

            transformed_states = self.prepare_transformed_states(parts_transformation, state)

            batch_paulis, batch_state = self.prepare_batch(diag_parts, transformed_states, return_cov or return_std)
            batch_expectation_values, batch_infos = self._ll_estimator.batch_estimate_paulis_on_state(
                batch_paulis, batch_state
            )

            if return_cov or return_std:
                parts_expectation_values, parts_covariances = self.extract_parts_expectation_values_and_covariances(
                    batch_expectation_values, parts_factors
                )
                paulis_covariances = self.assemble_paulis_covariances(parts_flat_idx, parts_covariances)
                if return_std:
                    parts_shots = np.array([infos["shots"] for infos in batch_infos])
                    paulis_shots = self.assemble_paulis_shots(parts_flat_idx, parts_shots)
            else:
                parts_expectation_values = self.extract_parts_expectation_values(
                    batch_expectation_values, parts_factors
                )

            paulis_expectation_values = self.assemble_paulis_expectation_values(
                parts_flat_idx, parts_expectation_values
            )

            pauli_obj_expectation_value = self._pauli_obj.expectation_values_from_paulis(paulis_expectation_values)

            out = (pauli_obj_expectation_value,)

            if return_cov:
                pauli_obj_covariance = self._pauli_obj.covariances_from_paulis(paulis_covariances)
                out += (pauli_obj_covariance,)

            if return_std:
                pauli_obj_standard_deviation = self._pauli_obj.standard_deviations_from_paulis(
                    paulis_covariances, paulis_shots
                )
                out += (pauli_obj_standard_deviation,)

            if len(out) == 1:
                return out[0]

            return out

        return NotImplemented
