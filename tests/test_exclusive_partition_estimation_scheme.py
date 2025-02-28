import time
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Estimator
from qiskit.primitives.backend_estimator_v2 import BackendEstimatorV2
from qiskit.primitives.backend_sampler_v2 import BackendSamplerV2
from qiskit.primitives.statevector_sampler import StatevectorSampler
from qiskit.quantum_info import pauli_basis
from qiskit_aer import AerSimulator

import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
import pauliarray.state.nqubit_state as nqs
from pauliarray.conversion.qiskit import pauli_array_to_pauli_list, weighted_pauli_array_from_pauli_list
from pauliarray.diagonalisation.commutating_paulis.with_operators import (
    general_to_diagonal as general_to_diagonal_with_operators,
)
from pauliarray.diagonalisation.commutating_paulis.with_qiskit_circuits import (
    general_to_diagonal as general_to_diagonal_with_qiskit_circuits,
)
from pauliarray.estimation.low_level.nqubit_state_estimators import NQubitStateDiagonalEstimator
from pauliarray.estimation.low_level.qiskit_interface_estimators import QiskitEstimatorWraper, QiskitSamplerEstimator
from pauliarray.estimation.low_level.statevector_estimators import StatevectorEstimator
from pauliarray.estimation.scheme.exclusive_partition_estimation import ExclusivePartitionEstimationScheme
from pauliarray.partition.commutating_paulis.exclusive_fct import partition_general_commutating, partition_same_x


class TestEstimationSchemeExclusivePartition(unittest.TestCase):

    def test_on_pauli_basis(self):

        num_qubits = 2

        state_circuit = random_circuit(num_qubits, 6)
        nqubit_state = nqs.NQubitState.from_qiskit_quantum_circuit(state_circuit)

        observable = weighted_pauli_array_from_pauli_list(pauli_list=pauli_basis(num_qubits))

        pauli_list = pauli_array_to_pauli_list(observable.paulis)

        estimator = Estimator()
        ref_result = estimator.run([state_circuit] * len(pauli_list), [pauli for pauli in pauli_list]).result()

        scheme_scenarios = [
            (
                NQubitStateDiagonalEstimator(),
                partition_same_x,
                general_to_diagonal_with_operators,
                nqubit_state,
            ),
            (
                QiskitEstimatorWraper(BackendEstimatorV2(backend=AerSimulator(shots=1e6))),
                partition_same_x,
                general_to_diagonal_with_qiskit_circuits,
                state_circuit,
            ),
        ]

        for scheme_scenario in scheme_scenarios:

            estimator, partition_fct, diag_fct, state = scheme_scenario

            estimation_scheme = ExclusivePartitionEstimationScheme(observable, estimator, partition_fct, diag_fct)

            observable_expectation_value = estimation_scheme.estimate_on_state(state)

            self.assertTrue(np.all(np.isclose(ref_result.values, observable_expectation_value, atol=1e-1)))

    def test_on_paulis_with_qiskit_circuits(self):

        paulis = pa.PauliArray.random((3, 5), 6)
        state_circuit = random_circuit(paulis.num_qubits, 6)
        nqubit_state = nqs.NQubitState.from_qiskit_quantum_circuit(state_circuit)

        nqubit_estimator = NQubitStateDiagonalEstimator()
        nqubit_scheme = ExclusivePartitionEstimationScheme(
            paulis, nqubit_estimator, partition_general_commutating, general_to_diagonal_with_qiskit_circuits
        )

        vector_estimator = StatevectorEstimator()
        vector_scheme = ExclusivePartitionEstimationScheme(
            paulis, vector_estimator, partition_general_commutating, general_to_diagonal_with_qiskit_circuits
        )
        num_shots = int(1e5)
        sampler_estimator = QiskitSamplerEstimator(BackendSamplerV2(backend=AerSimulator(shots=num_shots)))
        sampler_scheme = ExclusivePartitionEstimationScheme(
            paulis, sampler_estimator, partition_general_commutating, general_to_diagonal_with_qiskit_circuits
        )

        print()

        t0 = time.time()
        nqubit_paulis_expectation_value = nqubit_scheme.estimate_on_state(state_circuit)
        t1 = time.time()
        print("nqubit_scheme", t1 - t0)

        t0 = time.time()
        vector_paulis_expectation_value = vector_scheme.estimate_on_state(state_circuit)
        t1 = time.time()
        print("vector_scheme", t1 - t0)

        t0 = time.time()
        sampler_paulis_expectation_value = sampler_scheme.estimate_on_state(state_circuit)
        t1 = time.time()
        print("sampler_scheme", t1 - t0)

        print(sampler_paulis_expectation_value)

        t0 = time.time()
        ll_paulis_expectation_value = vector_estimator.estimate_paulis_on_state(paulis, state_circuit)
        t1 = time.time()
        print(t1 - t0)

        print()

        print("Paulis expectation values")

        print(
            np.stack(
                (
                    np.real(ll_paulis_expectation_value).flatten(),
                    np.real(nqubit_paulis_expectation_value).flatten(),
                    np.real(sampler_paulis_expectation_value).flatten(),
                )
            ).T
        )

        n_sigmas = 3 * float(1 / np.sqrt(num_shots))
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

        # paulis = pa.PauliArray.random((3, 1), 6)
        # state_circuit = random_circuit(paulis.num_qubits, 6)
        paulis = pa.PauliArray.from_labels(["XXXXXX", "XXXXXY", "ZZZZZZ"])
        state_circuit = QuantumCircuit(paulis.num_qubits)
        state_circuit.ry(np.pi / 3, 0)
        for q in range(1, paulis.num_qubits):
            state_circuit.cx(q - 1, q)
        state_circuit.x([0])

        n_shots = int(1e3)
        qiskit_statevector_sampler = StatevectorSampler(default_shots=n_shots)
        sampler_estimator = QiskitSamplerEstimator(qiskit_statevector_sampler)
        sampler_scheme = ExclusivePartitionEstimationScheme(
            paulis, sampler_estimator, partition_general_commutating, general_to_diagonal_with_qiskit_circuits
        )

        print()

        t0 = time.time()
        sampler_paulis_expectation_value, sampler_paulis_standard_deviations = (
            sampler_scheme.estimate_on_state_with_std(state_circuit)
        )

        t1 = time.time()
        print(t1 - t0)

        print()

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
