import time
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Sampler

import pauliarray.pauli.pauli_array as pa

# import pauliarray.pauli.pauli_operator as po
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.estimation.low_level.nqubit_state_estimators import NQubitStateEstimator
from pauliarray.estimation.low_level.statevector_estimators import StatevectorEstimator


class TestStatevectorEstimator(unittest.TestCase):
    def test_estimate_paulis_on_state_circuit(self):
        num_qubits = 8
        # paulis = gen_complete_pauli_array_basis(num_qubits).reshape((2**num_qubits, 2**num_qubits))
        paulis = pa.PauliArray.from_labels(["I" * num_qubits, "X" * num_qubits, "Y" * num_qubits, "Z" * num_qubits])

        state_circuit = QuantumCircuit(num_qubits)
        state_circuit.h(range(num_qubits))

        estimator = StatevectorEstimator()

        expectation_values = estimator.estimate_paulis_on_state(paulis, state_circuit)

        self.assertTrue(np.all(np.isclose(expectation_values, [1, 1, 0, 0])))


class TestNQubitStateEstimator(unittest.TestCase):
    def test_estimate_paulis_on_state_circuit(self):
        num_qubits = 8

        paulis = pa.PauliArray.random((2, 40), num_qubits)

        state_circuit = random_circuit(paulis.num_qubits, 6)

        estimator_0 = NQubitStateEstimator()
        estimator_1 = StatevectorEstimator()

        t0 = time.time()
        expectation_values_0 = estimator_0.estimate_paulis_on_state(paulis, state_circuit)
        t_0 = time.time() - t0
        print(t_0)
        t0 = time.time()
        expectation_values_1 = estimator_1.estimate_paulis_on_state(paulis, state_circuit)
        t_1 = time.time() - t0
        print(t_1)

        self.assertTrue(np.all(np.isclose(expectation_values_0, expectation_values_1)))
