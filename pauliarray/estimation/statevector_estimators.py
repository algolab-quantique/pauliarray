import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from pauliarray.estimation.base_estimators import BaseEstimator
from pauliarray.state.nqubit_state import NQubitState


class StatevectorEstimator(BaseEstimator):
    """
    Uses qiskit statevector simulator to compute expectation values of PauliArray.
    """

    def estimate_paulis_on_state_circuit(self, state_circuit: QuantumCircuit):
        """
        Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """
        state_circuit = state_circuit.copy()

        state = NQubitState.from_statevector(Statevector(state_circuit).data)

        paulis_expectation_values = state.pauli_array_expectation_values(self.paulis)

        paulis_covariances = np.zeros(self.paulis.shape + self.paulis.shape)

        return paulis_expectation_values, paulis_covariances


class SchemedStatevectorEstimator(BaseEstimator):

    def __init__(self, pauli_object, scheme):
        self._pauli_object = pauli_object
        self._scheme = scheme
