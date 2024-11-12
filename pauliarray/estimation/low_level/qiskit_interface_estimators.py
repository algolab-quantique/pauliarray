from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
from qiskit.quantum_info import Statevector

import pauliarray.state.basis_state_array as bsa
from pauliarray.conversion.qiskit import pauli_array_to_pauli_list
from pauliarray.estimation.base_estimators import DiagonalEstimator, GeneralEstimator
from pauliarray.pauli.pauli_array import PauliArray
from pauliarray.state.nqubit_state import NQubitState


class QiskitSamplerEstimator(DiagonalEstimator):

    def __init__(self, sampler: BaseSamplerV2):

        self._qiskit_sampler = sampler

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

        sampler = self._qiskit_sampler

        state_circuit = state_circuit.copy()
        state_circuit.measure_all()

        job = sampler.run([state_circuit])
        result = job.result()[0]

        meas = result.data.meas

        bit_strings = np.unpackbits(meas.array, axis=-1, bitorder="little", count=meas.num_bits).astype(bool)

        meas_states = bsa.BasisStateArray(bit_strings)
        basis_states, counts = bsa.fast_flat_unique(meas_states, return_counts=True)
        nqubit_state = NQubitState(basis_states, np.sqrt(counts)).normalise()
        paulis_expectation_values = nqubit_state.pauli_array_expectation_values(paulis)

        return paulis_expectation_values


class QiskitEstimatorWraper(GeneralEstimator):
    def __init__(self, estimator: BaseEstimatorV2):

        self._qiskit_estimator = estimator

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
        pauli_list = pauli_array_to_pauli_list(paulis.flatten())

        estimator = self._qiskit_estimator

        job = estimator.run([(state_circuit, pauli_list)])
        result = job.result()[0]

        evs = result.data.evs

        paulis_expectation_values = evs.reshape(paulis.shape)

        return paulis_expectation_values
