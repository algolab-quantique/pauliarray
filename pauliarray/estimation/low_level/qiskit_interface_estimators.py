from typing import Any, List, Tuple

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

        sampler = self._qiskit_sampler

        all_circuits = []
        for state_circuit in batch_state:
            state_circuit = state_circuit.copy()
            state_circuit.measure_all()
            all_circuits.append(state_circuit)

        job = sampler.run(all_circuits)
        results = job.result()
        batch_expectation_values = []
        batch_infos = []
        for paulis, result in zip(batch_paulis, results):
            meas = result.data.meas

            bit_strings = np.unpackbits(meas.array, axis=-1, bitorder="little", count=meas.num_bits).astype(bool)

            meas_states = bsa.BasisStateArray(bit_strings)
            basis_states, counts = bsa.fast_flat_unique(meas_states, return_counts=True)
            nqubit_state = NQubitState(basis_states, np.sqrt(counts)).normalise()
            paulis_expectation_values = nqubit_state.pauli_array_expectation_values(paulis)
            batch_expectation_values.append(paulis_expectation_values)
            batch_infos.append({"shots": result.metadata["shots"]})

        return batch_expectation_values, batch_infos


class QiskitEstimatorWraper(GeneralEstimator):
    def __init__(self, estimator: BaseEstimatorV2):

        self._qiskit_estimator = estimator

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

        estimator = self._qiskit_estimator

        pubs = []
        for paulis, state_circuit in zip(batch_paulis, batch_state):
            pauli_list = pauli_array_to_pauli_list(paulis.flatten())
            pubs.append((state_circuit, pauli_list))

        job = estimator.run(pubs)
        results = job.result()

        batch_expectation_values = []
        batch_infos = []
        for paulis, result in zip(batch_paulis, results):
            evs = result.data.evs
            paulis_expectation_values = evs.reshape(paulis.shape)
            batch_expectation_values.append(paulis_expectation_values)
            batch_infos.append({})

        return batch_expectation_values, batch_infos
