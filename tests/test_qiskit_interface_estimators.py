import time
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Sampler
from qiskit.primitives.statevector_estimator import StatevectorEstimator as QiskitStatevectorEstimator
from qiskit.primitives.statevector_sampler import StatevectorSampler

import pauliarray.pauli.pauli_array as pa

# import pauliarray.pauli.pauli_operator as po
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.estimation.low_level.nqubit_state_estimators import NQubitStateEstimator
from pauliarray.estimation.low_level.qiskit_interface_estimators import QiskitEstimatorWraper, QiskitSamplerEstimator
from pauliarray.estimation.low_level.statevector_estimators import StatevectorEstimator


class TestQiskitSamplerEstimator(unittest.TestCase):
    def test_estimate_paulis_on_state_circuit(self):
        num_qubits = 8

        n_shots = int(1e5)

        paulis = pa.PauliArray.random((2, 10, 4), num_qubits, diagonal=True)

        state_circuit = random_circuit(paulis.num_qubits, 6)

        vector_estimator = StatevectorEstimator()
        sampler_estimator = QiskitSamplerEstimator(StatevectorSampler(default_shots=n_shots))

        t0 = time.time()
        vector_expectation_values = np.real(vector_estimator.estimate_paulis_on_state(paulis, state_circuit))
        t_0 = time.time() - t0
        print(t_0)
        t0 = time.time()
        sampler_expectation_values = sampler_estimator.estimate_paulis_on_state(paulis, state_circuit)
        t_1 = time.time() - t0
        print(t_1)

        n_sigmas = 10 / np.sqrt(n_shots)

        self.assertTrue(np.all(np.abs(vector_expectation_values - sampler_expectation_values) < n_sigmas))


class TestQiskitEstimatorWraper(unittest.TestCase):
    def test_estimate_paulis_on_state_circuit(self):
        num_qubits = 8

        n_shots = int(1e5)

        paulis = pa.PauliArray.random((2, 10, 4), num_qubits, diagonal=False)

        state_circuit = random_circuit(paulis.num_qubits, 6)

        vector_estimator = StatevectorEstimator()
        estimator_wraper = QiskitEstimatorWraper(QiskitStatevectorEstimator())

        t0 = time.time()
        vector_expectation_values = np.real(vector_estimator.estimate_paulis_on_state(paulis, state_circuit))
        t_0 = time.time() - t0
        print(t_0)
        t0 = time.time()
        sampler_expectation_values = estimator_wraper.estimate_paulis_on_state(paulis, state_circuit)
        t_1 = time.time() - t0
        print(t_1)

        n_sigmas = 10 / np.sqrt(n_shots)
        # print(n_sigmas)

        # print(
        #     np.stack(
        #         (
        #             np.real(vector_expectation_values).flatten(),
        #             np.real(nqubit_expectation_values).flatten(),
        #             np.real(sampler_expectation_values).flatten(),
        #         )
        #     ).T
        # )

        self.assertTrue(np.all(np.abs(vector_expectation_values - sampler_expectation_values) < n_sigmas))
