import abc
from typing import Any, Callable, List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

import pauliarray.pauli.pauli_array as pa
import pauliarray.state.basis_state_array as bsa
from pauliarray.pauli.pauli_array import PauliArray

# from qiskit.primitives import Sampler

# import pauliarray.state.qubit_state as qbs


class BaseEstimator(object):
    def estimate_paulis_on_state(self, paulis: PauliArray, state: Any, return_infos=False):

        out = self.batch_estimate_paulis_on_state([paulis], [state], return_infos)

        if return_infos:
            return out[0][0], out[1][0]

        return out

    def batch_estimate_paulis_on_state(
        self, batch_paulis: List[PauliArray], batch_state: List[Any], return_infos=False
    ):
        return NotImplemented


class DiagonalEstimator(BaseEstimator):
    pass


class BitwiseEstimator(DiagonalEstimator):
    pass


class GeneralEstimator(BitwiseEstimator):
    pass
