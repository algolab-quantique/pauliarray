from typing import Any

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

    def estimate_paulis_on_state_circuit(self, paulis: PauliArray, state_circuit: QuantumCircuit):
        """
        Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """
        state_circuit = state_circuit.copy()

        state = NQubitState.from_statevector(Statevector(state_circuit).data)

        paulis_expectation_values = state.pauli_array_expectation_values(paulis)

        # paulis_covariances = np.zeros(paulis.shape + paulis.shape)

        return paulis_expectation_values


class NQubitStateDiagonalEstimator(DiagonalEstimator):
    """
    Uses qiskit statevector simulator to compute expectation values of PauliArray.
    """

    def estimate_paulis_on_state(self, paulis: PauliArray, state: Any):

        if isinstance(state, QuantumCircuit):
            return self.estimate_paulis_on_state_circuit(paulis, state)

    def estimate_paulis_on_state_circuit(self, paulis: PauliArray, state_circuit: QuantumCircuit):
        """
        Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """
        assert np.all(paulis.is_diagonal())

        state_circuit = state_circuit.copy()

        state = NQubitState.from_statevector(Statevector(state_circuit).data)

        paulis_expectation_values = state.diagonal_pauli_array_expectation_values(paulis)

        return paulis_expectation_values
