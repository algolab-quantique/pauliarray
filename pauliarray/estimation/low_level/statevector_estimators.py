from typing import Any, List

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

    def batch_estimate_paulis_on_state(
        self, batch_paulis: List[PauliArray], batch_state: List[Any], return_infos=False
    ):

        assert np.all([isinstance(state, type(batch_state[0])) for state in batch_state])

        if isinstance(batch_state[0], QuantumCircuit):
            return self.batch_estimate_paulis_on_state_circuit(batch_paulis, batch_state, return_infos)

        return NotImplemented

    def batch_estimate_paulis_on_state_circuit(
        self, batch_paulis: List[PauliArray], batch_state: List[QuantumCircuit], return_infos=False
    ):
        """
        Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """

        batch_expectation_values = []
        batch_infos = []
        for paulis, state_circuit in zip(batch_paulis, batch_state):
            statevector = Statevector(state_circuit).data
            matrices = paulis.to_matrices()
            paulis_expectation_values = np.einsum("i,j,...ij->...", np.conj(statevector), statevector, matrices)
            batch_expectation_values.append(paulis_expectation_values)
            batch_infos.append({})

        if return_infos:
            return batch_expectation_values, batch_infos

        return batch_expectation_values
