from typing import List, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from qiskit import qasm3
from qiskit.circuit import QuantumCircuit

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_circuits import (
    general_to_diagonal as general_to_diagonal_with_circuits,
)


def general_to_diagonal(
    paulis: pa.PauliArray, force_single_qubit_generators=False
) -> Tuple[pa.PauliArray, NDArray[np.complex128], QuantumCircuit]:
    """
    Converts a 1D PauliArray of commuting Pauli strings into bitwise commuting pauli strings and factors. Also returns the QuantumCircuit which performs the conversion.

    Args:
        paulis (PauliArray): 1D PauliArray of commuting Pauli strings
        force_single_qubit_generators(bool): For already bitwise commuting qubits, the transformation will apply a single qubit rotation to make it diagonal. This prevents some unnecessary n-qubits rotations.

    Returns:
        PauliArray: 1D PauliArray of bitwise commuting Pauli strings
        NDArray[np.complex128]: Phase factors resulting from the transformation
        QuantumCircuit: The transformation given as a qiskit QuantumCircuit
    """

    (diag_paulis, factors), qasm_circuit_str = general_to_diagonal_with_circuits(paulis, force_single_qubit_generators)

    return (diag_paulis, factors), qasm3.loads(qasm_circuit_str)
