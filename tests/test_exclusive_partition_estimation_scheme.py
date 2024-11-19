import time
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.primitives.statevector_sampler import StatevectorSampler

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_circuits import diagonalise_with_circuits
from pauliarray.estimation.low_level.nqubit_state_estimators import NQubitStateDiagonalEstimator
from pauliarray.estimation.low_level.qiskit_interface_estimators import QiskitSamplerEstimator
from pauliarray.estimation.low_level.statevector_estimators import StatevectorEstimator
from pauliarray.estimation.scheme.exclusive_partition_estimation import ExclusivePartitionEstimationScheme
from pauliarray.partition.commutating_paulis.exclusive_fct import partition_general_commutating, partition_same_x


class TestEstimationSchemeExclusivePartition(unittest.TestCase):
    def test_on_paulis(self):

        paulis = pa.PauliArray.random((3, 5), 6)
        state_circuit = random_circuit(paulis.num_qubits, 6)
        # paulis = pa.PauliArray.from_labels(["XXXXXX", "XXYXXY", "ZZZZZZ"])
        # state_circuit = QuantumCircuit(paulis.num_qubits)
        # state_circuit.h(0)
        # for q in range(1, paulis.num_qubits):
        #     state_circuit.cx(q - 1, q)
        # state_circuit.x([0])

        nqubit_estimator = NQubitStateDiagonalEstimator()
        nqubit_scheme = ExclusivePartitionEstimationScheme(
            paulis, nqubit_estimator, partition_general_commutating, diagonalise_with_circuits
        )

        vector_estimator = StatevectorEstimator()
        vector_scheme = ExclusivePartitionEstimationScheme(
            paulis, vector_estimator, partition_general_commutating, diagonalise_with_circuits
        )
        n_shots = int(1e5)
        sampler_estimator = QiskitSamplerEstimator(StatevectorSampler(default_shots=n_shots))
        sampler_scheme = ExclusivePartitionEstimationScheme(
            paulis, sampler_estimator, partition_general_commutating, diagonalise_with_circuits
        )

        print()

        t0 = time.time()
        nqubit_paulis_expectation_value = nqubit_scheme.estimate_on_state(state_circuit)
        t1 = time.time()
        print(t1 - t0)

        t0 = time.time()
        vector_paulis_expectation_value = vector_scheme.estimate_on_state(state_circuit)
        t1 = time.time()
        print(t1 - t0)

        t0 = time.time()
        sampler_paulis_expectation_value = sampler_scheme.estimate_on_state(state_circuit)
        t1 = time.time()
        print(t1 - t0)

        print(sampler_paulis_expectation_value)

        t0 = time.time()
        ll_paulis_expectation_value = vector_estimator.estimate_paulis_on_state(paulis, state_circuit)
        t1 = time.time()
        print(t1 - t0)

        print()

        # print(nqubit_paulis_expectation_value)
        # print(vector_paulis_expectation_value)
        # print(ll_paulis_expectation_value)

        print(
            np.stack(
                (
                    np.real(ll_paulis_expectation_value).flatten(),
                    np.real(nqubit_paulis_expectation_value).flatten(),
                    np.real(sampler_paulis_expectation_value).flatten(),
                )
            ).T
        )

        n_sigmas = float(10 / np.sqrt(n_shots))
        print(f"{n_sigmas=}")

        self.assertTrue(np.all(np.isclose(nqubit_paulis_expectation_value, ll_paulis_expectation_value)))
        self.assertTrue(np.all(np.isclose(vector_paulis_expectation_value, ll_paulis_expectation_value)))

        # print(np.abs(sampler_paulis_expectation_value - ll_paulis_expectation_value))
        self.assertTrue(np.all(np.abs(sampler_paulis_expectation_value - ll_paulis_expectation_value) < n_sigmas))

        # for part in scheme.diag_parts:
        #     print(part.inspect())

        # for circuit in scheme.parts_transformation:
        #     print(circuit)

    def test_on_paulis_with_cov(self):

        # paulis = pa.PauliArray.random((3, 1), 6)
        # state_circuit = random_circuit(paulis.num_qubits, 6)
        paulis = pa.PauliArray.from_labels(["XXXXXX", "XXXXXY", "ZZZZZZ"])
        state_circuit = QuantumCircuit(paulis.num_qubits)
        state_circuit.ry(np.pi / 3, 0)
        for q in range(1, paulis.num_qubits):
            state_circuit.cx(q - 1, q)
        state_circuit.x([0])

        nqubit_estimator = NQubitStateDiagonalEstimator()
        nqubit_scheme = ExclusivePartitionEstimationScheme(
            paulis, nqubit_estimator, partition_general_commutating, diagonalise_with_circuits
        )

        vector_estimator = StatevectorEstimator()
        vector_scheme = ExclusivePartitionEstimationScheme(
            paulis, vector_estimator, partition_general_commutating, diagonalise_with_circuits
        )
        n_shots = int(1e3)
        sampler_estimator = QiskitSamplerEstimator(StatevectorSampler(default_shots=n_shots))
        sampler_scheme = ExclusivePartitionEstimationScheme(
            paulis, sampler_estimator, partition_general_commutating, diagonalise_with_circuits
        )

        print()

        t0 = time.time()
        nqubit_paulis_expectation_value, nqubit_paulis_covariances = nqubit_scheme.estimate_on_state(
            state_circuit, return_cov=True
        )
        t1 = time.time()
        print(t1 - t0)

        t0 = time.time()
        vector_paulis_expectation_value, vector_paulis_covariances = vector_scheme.estimate_on_state(
            state_circuit, return_cov=True
        )
        t1 = time.time()
        print(t1 - t0)

        t0 = time.time()
        sampler_paulis_expectation_value, sampler_paulis_covariances = sampler_scheme.estimate_on_state(
            state_circuit, return_cov=True
        )
        t1 = time.time()
        print(t1 - t0)

        t0 = time.time()
        ll_paulis_expectation_value = vector_estimator.estimate_paulis_on_state(paulis, state_circuit)
        t1 = time.time()
        print(t1 - t0)

        print()

        # print(nqubit_paulis_expectation_value)
        # print(vector_paulis_expectation_value)
        # print(ll_paulis_expectation_value)

        print(
            np.stack(
                (
                    np.real(ll_paulis_expectation_value).flatten(),
                    np.real(nqubit_paulis_expectation_value).flatten(),
                    np.real(sampler_paulis_expectation_value).flatten(),
                )
            ).T
        )

        print(nqubit_paulis_covariances / n_shots)
        print(vector_paulis_covariances / n_shots)
        print(sampler_paulis_covariances / n_shots)

        n_sigmas = float(10 / np.sqrt(n_shots))
        print(f"{n_sigmas=}")

        self.assertTrue(np.all(np.isclose(nqubit_paulis_expectation_value, ll_paulis_expectation_value)))
        self.assertTrue(np.all(np.isclose(vector_paulis_expectation_value, ll_paulis_expectation_value)))

        # print(np.abs(sampler_paulis_expectation_value - ll_paulis_expectation_value))
        self.assertTrue(np.all(np.abs(sampler_paulis_expectation_value - ll_paulis_expectation_value) < n_sigmas))

        # for part in scheme.diag_parts:
        #     print(part.inspect())

        # for circuit in scheme.parts_transformation:
        #     print(circuit)

    def test_on_paulis_with_std(self):

        paulis = pa.PauliArray.random((3, 1), 6)
        state_circuit = random_circuit(paulis.num_qubits, 6)
        # paulis = pa.PauliArray.from_labels(["XXXXXX", "XXXXXY", "ZZZZZZ"])
        # state_circuit = QuantumCircuit(paulis.num_qubits)
        # state_circuit.ry(np.pi / 3, 0)
        # for q in range(1, paulis.num_qubits):
        #     state_circuit.cx(q - 1, q)
        # state_circuit.x([0])

        n_shots = int(1e3)
        qiskit_statevector_sampler = StatevectorSampler(default_shots=n_shots)
        sampler_estimator = QiskitSamplerEstimator(qiskit_statevector_sampler)
        sampler_scheme = ExclusivePartitionEstimationScheme(
            paulis, sampler_estimator, partition_general_commutating, diagonalise_with_circuits
        )

        print()

        t0 = time.time()
        sampler_paulis_expectation_value, sampler_paulis_standard_deviations = sampler_scheme.estimate_on_state(
            state_circuit, return_std=True
        )
        t1 = time.time()
        print(t1 - t0)

        print()

        # print(nqubit_paulis_expectation_value)
        # print(vector_paulis_expectation_value)
        # print(ll_paulis_expectation_value)

        print(
            np.stack(
                (
                    np.real(sampler_paulis_expectation_value).flatten(),
                    np.real(sampler_paulis_standard_deviations).flatten(),
                )
            ).T
        )

        # print(nqubit_paulis_covariances / n_shots)
        # print(vector_paulis_covariances / n_shots)
        # print(sampler_paulis_covariances / n_shots)

        # n_sigmas = float(10 / np.sqrt(n_shots))
        # print(f"{n_sigmas=}")

        # self.assertTrue(np.all(np.isclose(nqubit_paulis_expectation_value, ll_paulis_expectation_value)))
        # self.assertTrue(np.all(np.isclose(vector_paulis_expectation_value, ll_paulis_expectation_value)))

        # # print(np.abs(sampler_paulis_expectation_value - ll_paulis_expectation_value))
        # self.assertTrue(np.all(np.abs(sampler_paulis_expectation_value - ll_paulis_expectation_value) < n_sigmas))

        # # for part in scheme.diag_parts:
        #     print(part.inspect())

        # for circuit in scheme.parts_transformation:
        #     print(circuit)

    # def test_h2(self):
