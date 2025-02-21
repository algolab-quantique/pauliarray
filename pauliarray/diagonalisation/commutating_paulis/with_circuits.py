from typing import List, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from qiskit import qasm3

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_qiskit_circuits import (
    general_to_diagonal as general_to_diagonal_with_qiskit_circuits,
)
from pauliarray.utils.protocols import HasPaulis


def general_to_diagonal(
    paulis: pa.PauliArray, force_single_qubit_generators=False
) -> Tuple[pa.PauliArray, NDArray[np.complex128], str]:
    """
    Converts a 1D PauliArray of commuting Pauli strings into bitwise commuting pauli strings and factors. Also returns the OpenQasm3 circuit which performs the conversion.

    Args:
        paulis (PauliArray): 1D PauliArray of commuting Pauli strings
        force_single_qubit_generators(bool): For already bitwise commuting qubits, the transformation will apply a single qubit rotation to make it diagonal. This prevents some unnecessary n-qubits rotations.

    Returns:
        PauliArray: 1D PauliArray of bitwise commuting Pauli strings
        NDArray[np.complex128]: Phase factors resulting from the transformation
        str: The transformation given as a OpemQasm3 quantum circuit
    """

    (diag_paulis, factors), circuit = general_to_diagonal_with_qiskit_circuits(
        paulis, force_single_qubit_generators=force_single_qubit_generators
    )

    return (diag_paulis, factors), qasm3.dumps(circuit)
