import abc
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

import pauliarray.pauli.pauli_array as pa
import pauliarray.state.basis_state_array as bsa
from pauliarray.pauli.pauli_array import PauliArray

# from qiskit.primitives import Sampler

# import pauliarray.state.qubit_state as qbs


class BaseEstimator(object):
    def estimate_paulis_on_state(self, paulis: PauliArray, state: Any):
        pass


class DiagonalEstimator(BaseEstimator):
    pass


class BitwiseEstimator(DiagonalEstimator):
    pass


class GeneralEstimator(BitwiseEstimator):
    pass


# class DiagonalToGeneralEstimator(GeneralEstimator):
#     def __init__(self, diagonal_estimator: DiagonalEstimator, diagonalisation_fct: Callable):
#         self._diagonal_estimator = diagonal_estimator
#         self._diagonalisation_fct = diagonalisation_fct

#     def estimate_paulis_on_state_circuit(self, paulis: PauliArray, state_circuit: QuantumCircuit):
#         """
#         Estimate the expectation value of the paulis using the statevector simulator of Qiskit.

#         Args:
#             state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

#         Returns:
#             NDArray: _description_
#         """
#         state_circuit = state_circuit.copy()

#         state = NQubitState.from_statevector(Statevector(state_circuit).data)

#         paulis_expectation_values = state.pauli_array_expectation_values(paulis)

#         paulis_covariances = np.zeros(paulis.shape + paulis.shape)

#         return paulis_expectation_values, paulis_covariances
