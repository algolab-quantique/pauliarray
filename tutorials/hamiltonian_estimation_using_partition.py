# %%

import time
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Estimator, Sampler
from qiskit.primitives.backend_estimator_v2 import BackendEstimatorV2
from qiskit.primitives.backend_sampler_v2 import BackendSamplerV2
from qiskit.primitives.statevector_sampler import StatevectorSampler
from qiskit.quantum_info import Statevector, pauli_basis
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.drivers import PySCFDriver

import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.conversion.qiskit import extract_fermionic_op, weighted_pauli_array_from_pauli_list
from pauliarray.diagonalisation.commutating_paulis.with_circuits import (
    general_to_diagonal as general_to_diagonal_with_circuit,
)
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
from pauliarray.mapping.fermion import BravyiKitaev, JordanWigner, Parity
from pauliarray.partition.commutating_paulis.exclusive_fct import (
    partition_bitwise_commutating,
    partition_general_commutating,
    partition_same_x,
    partition_same_x_plus_special,
)
from pauliarray.state import nqubit_state as nqs

# In this tutorial we will see how PauliArray can be used to estimate the espectation value of an Hamiltonian by partionning it and diagonalizing its parts

# %%

mol_info = {
    # "atom": "N 0 0 -0.545;N 0 0 0.545;",
    "atom": "Li 0 0 0;H 0 0 1.6;",
    # "atom": "H 0 0 0;H 0 0 0.735;",
    "basis": "sto3g",
    "charge": 0,
    "spin": 0,
}

driver = PySCFDriver(**mol_info)
problem = driver.run()
hamiltonian = problem.hamiltonian
second_q_hamiltonian = problem.hamiltonian.second_q_op()

num_spin_orbitals = second_q_hamiltonian.num_spin_orbitals

one_body_tuple, two_body_tuple = extract_fermionic_op(second_q_hamiltonian)

mapping = JordanWigner(num_spin_orbitals)
qubit_hamiltonian = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)


# %%

observable = qubit_hamiltonian
state_circuit = random_circuit(qubit_hamiltonian.num_qubits, 6)


# %%

# QiskitSamplerEstimator(BackendSamplerV2(backend=AerSimulator(shots=1e6))),

num_shots = 1e6

scheme_scenarios = [
    # (
    #     QiskitEstimatorWraper(BackendEstimatorV2(backend=AerSimulator(shots=num_shots ))),
    #     partition_bitwise_commutating,
    #     general_to_diagonal_with_operators,
    #     state_circuit,
    # ),
    (
        QiskitEstimatorWraper(BackendEstimatorV2(backend=AerSimulator(shots=num_shots))),
        partition_general_commutating,
        general_to_diagonal_with_qiskit_circuits,
        state_circuit,
    ),
    (
        QiskitEstimatorWraper(BackendEstimatorV2(backend=AerSimulator(shots=num_shots))),
        partition_same_x,
        general_to_diagonal_with_qiskit_circuits,
        state_circuit,
    ),
    (
        QiskitEstimatorWraper(BackendEstimatorV2(backend=AerSimulator(shots=num_shots))),
        partition_same_x_plus_special,
        general_to_diagonal_with_qiskit_circuits,
        state_circuit,
    ),
    # (
    #     QiskitEstimatorWraper(BackendEstimatorV2(backend=AerSimulator(shots=num_shots ))),
    #     partition_same_x,
    #     general_to_diagonal_with_qiskit_circuits,
    #     state_circuit,
    # ),
]


for scheme_scenario in scheme_scenarios:

    estimator, partition_fct, diag_fct, state = scheme_scenario

    t0 = time.time()

    estimation_scheme = ExclusivePartitionEstimationScheme(observable, estimator, partition_fct, diag_fct)

    t1 = time.time()

    hamiltonian_expectation_value = estimation_scheme.estimate_on_state(state)

    t2 = time.time()

    print(f"{type(estimator).__name__:30s}, {partition_fct.__name__:30s}, {t1 - t0:.3f}, {t2 - t1:.3f}, {t2 - t0:.3f}")

    # print(hamiltonian_expectation_value)

# %%
