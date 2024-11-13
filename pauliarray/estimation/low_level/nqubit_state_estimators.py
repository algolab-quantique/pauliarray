from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from pauliarray.estimation.base_estimators import DiagonalEstimator, GeneralEstimator
from pauliarray.pauli.pauli_array import PauliArray
from pauliarray.state.nqubit_state import NQubitState


class NQubitStateEstimator(GeneralEstimator):
    """
    Uses qiskit statevector simulator to compute expectation values of PauliArray.
    """

    def batch_estimate_paulis_on_state(self, batch_paulis: List[PauliArray], batch_state: List[Any]):

        assert np.all([isinstance(state, type(batch_state[0])) for state in batch_state])

        if isinstance(batch_state[0], QuantumCircuit):
            return self.batch_estimate_paulis_on_state_circuit(batch_paulis, batch_state)

        return NotImplemented

    def batch_estimate_paulis_on_state_circuit(self, batch_paulis: List[PauliArray], batch_state: List[QuantumCircuit]):
        """
        Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """

        batch_expectation_values = []
        for paulis, state_circuit in zip(batch_paulis, batch_state):
            state = NQubitState.from_statevector(Statevector(state_circuit).data)
            paulis_expectation_values = state.pauli_array_expectation_values(paulis)
            batch_expectation_values.append(paulis_expectation_values)

        return batch_expectation_values


class NQubitStateDiagonalEstimator(DiagonalEstimator):
    """
    Uses qiskit statevector simulator to compute expectation values of PauliArray.
    """

    def batch_estimate_paulis_on_state(self, batch_paulis: List[PauliArray], batch_state: List[Any]):

        assert np.all([isinstance(state, type(batch_state[0])) for state in batch_state])

        if isinstance(batch_state[0], QuantumCircuit):
            return self.batch_estimate_paulis_on_state_circuit(batch_paulis, batch_state)

        return NotImplemented

    def batch_estimate_paulis_on_state_circuit(self, batch_paulis: List[PauliArray], batch_state: List[QuantumCircuit]):
        """
        Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """
        assert np.all([np.all(paulis.is_diagonal()) for paulis in batch_paulis])

        batch_expectation_values = []
        for paulis, state_circuit in zip(batch_paulis, batch_state):
            state = NQubitState.from_statevector(Statevector(state_circuit).data)
            paulis_expectation_values = state.diagonal_pauli_array_expectation_values(paulis)
            batch_expectation_values.append(paulis_expectation_values)

        return batch_expectation_values
