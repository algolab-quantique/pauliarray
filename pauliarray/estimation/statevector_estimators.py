import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from pauliarray.estimation.base_estimators import BaseEstimator


class StateVectorEstimator(BaseEstimator):
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

        statevector = Statevector(state_circuit).data

        # print(f"{statevector=}")

        matrices = self.paulis.to_matrices()

        # all_matrices = np.zeros(matrices.shape + (len(statevector), len(statevector)), dtype=complex)

        # print(f"{all_matrices.shape=}")

        # for idx in np.ndindex(matrices.shape):
        #     all_matrices[idx, :, :] = matrices[idx]

        paulis_expectation_values = np.zeros(matrices.shape, dtype=complex)
        for idx in np.ndindex(matrices.shape):

            paulis_expectation_values[idx] = np.einsum("i,j,ij->...", np.conj(statevector), statevector, matrices[idx])

        paulis_covariances = np.zeros(self.paulis.shape + self.paulis.shape)

        return paulis_expectation_values, paulis_covariances
