from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import pauliarray.state.basis_state_array as bsa
from pauliarray.conversion.qiskit import pauli_array_to_pauli_list
from pauliarray.estimation.base_estimators import DiagonalEstimator, GeneralEstimator
from pauliarray.pauli.pauli_array import PauliArray
from pauliarray.state.nqubit_state import NQubitState


class QiskitSamplerEstimator(DiagonalEstimator):

    def __init__(self, sampler: BaseSamplerV2):

        self._qiskit_sampler = sampler

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

        sampler = self._qiskit_sampler

        all_circuits = []
        for state_circuit in batch_state:
            state_circuit = state_circuit.copy()
            state_circuit.measure_all()
            all_circuits.append(state_circuit)

        if hasattr(sampler, "backend"):
            pass_manager = generate_preset_pass_manager(backend=sampler.backend, optimization_level=1)
            isa_circuits = pass_manager.run(all_circuits)
        else:
            isa_circuits = all_circuits

        job = sampler.run(isa_circuits)
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

        if return_infos:
            return batch_expectation_values, batch_infos

        return batch_expectation_values


class QiskitEstimatorWraper(GeneralEstimator):
    def __init__(self, estimator: BaseEstimatorV2):

        self._qiskit_estimator = estimator

    def batch_estimate_paulis_on_state(
        self, batch_paulis: List[PauliArray], batch_state: List[Any], return_infos=False
    ):

        assert np.all([isinstance(state, type(batch_state[0])) for state in batch_state])

        if isinstance(batch_state[0], QuantumCircuit):
            return self.batch_estimate_paulis_on_state_circuit(batch_paulis, batch_state, return_infos)

        return NotImplemented

    def batch_estimate_paulis_on_state_circuit(
        self, batch_paulis: List[PauliArray], batch_state_circuits: List[QuantumCircuit], return_infos=False
    ):
        """
        Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """

        estimator = self._qiskit_estimator

        if hasattr(estimator, "backend"):
            pass_manager = generate_preset_pass_manager(backend=estimator.backend, optimization_level=1)
            isa_batch_state_circuits = pass_manager.run(batch_state_circuits)
        else:
            isa_batch_state_circuits = batch_state_circuits

        pubs = []
        for paulis, isa_state_circuit in zip(batch_paulis, isa_batch_state_circuits):
            pubs.append((isa_state_circuit, pauli_array_to_pauli_list(paulis.flatten())))

        job = estimator.run(pubs)
        results = job.result()

        batch_expectation_values = []
        batch_infos = []
        for paulis, result in zip(batch_paulis, results):
            evs = result.data.evs
            paulis_expectation_values = evs.reshape(paulis.shape)
            batch_expectation_values.append(paulis_expectation_values)
            batch_infos.append({})

        if return_infos:
            return batch_expectation_values, batch_infos

        return batch_expectation_values
