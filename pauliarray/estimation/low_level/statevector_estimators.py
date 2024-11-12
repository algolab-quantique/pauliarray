from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from pauliarray.estimation.base_estimators import GeneralEstimator
from pauliarray.pauli.pauli_array import PauliArray
from pauliarray.state.nqubit_state import NQubitState


class StatevectorEstimator(GeneralEstimator):
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
        state_circuit = state_circuit.copy()

        statevector = Statevector(state_circuit).data

        matrices = paulis.to_matrices()

        # paulis_expectation_values = np.zeros(matrices.shape[:-2], dtype=complex)
        # for idx in np.ndindex(matrices.shape[:-2]):
        #     paulis_expectation_values[idx] = np.einsum("i,j,ij->...", np.conj(statevector), statevector, matrices[idx])

        paulis_expectation_values = np.einsum("i,j,...ij->...", np.conj(statevector), statevector, matrices)

        return paulis_expectation_values
