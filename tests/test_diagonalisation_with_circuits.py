import unittest

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_qiskit_circuits import general_to_diagonal

cases_paulis = [
    pa.PauliArray.from_labels(
        [
            "IIIZ",
            "IIZZ",
            "IZZZ",
            "ZZZZ",
        ]
    ),
    pa.PauliArray.from_labels(
        [
            "XXXX",
            "XXYY",
            "YYXX",
            "YYYY",
        ]
    ),
    pa.PauliArray.from_labels(
        [
            "ZIIIIXII",
            "ZIIIIIXX",
            "IIIIIXYY",
            "ZZZXIXII",
            "IIIIXXII",
            "IZXZXXII",
            "IXZZXXII",
            "ZIIIIIYY",
            "ZZZXIIYY",
            "ZZZXXIII",
            "IIIIIIZZ",
            "IYIYIIZZ",
            "ZZZXIIII",
            "ZZZXIIXX",
            "IIIIXIYY",
            "ZIIIIIII",
            "IIIIIIYY",
            "IIIIXIXX",
            "IZZXXXZZ",
            "ZIIIXIII",
            "IZZXIIII",
            "IIYYIIZZ",
            "IIIIXIII",
            "IYYIIIII",
        ]
    ),
    pa.PauliArray.from_labels(
        [
            "XZXXZZXII",
            "YYIXZZXII",
            "IYYXZZXII",
            "XZXYZZYII",
            "YYIYZZYII",
            "IYYYZZYII",
            "XZXIIIIZI",
            "YYIIIIIZI",
            "IYYIIIIZI",
            "XZXIIIIIZ",
            "YYIIIIIIZ",
            "IYYIIIIIZ",
        ]
    ),
    # pa.PauliArray.from_labels(
    #     [
    #         "XZZZZXXZZZZX",
    #         "YZZZZYXZZZZX",
    #         "XZZZZXYZZZZY",
    #         "YZZZZYYZZZZY",
    #     ]
    # ),
]


def assert_unitaries_equivalent(unitary_matrix_1, unitary_matrix_2):

    test_phase_id_matrix = unitary_matrix_1.T.conj() @ unitary_matrix_2

    assert np.all(
        np.isclose(test_phase_id_matrix.T.conj() @ test_phase_id_matrix, np.eye(test_phase_id_matrix.shape[0]))
    )


class TestDiagonalisationWithCircuits(unittest.TestCase):

    def test_general_to_diagonal(self):

        for paulis in cases_paulis:

            (diag_paulis, factors), circuit = general_to_diagonal(paulis, force_single_qubit_generators=True)

            paulis_matrices = paulis.to_matrices()
            diag_paulis_matrices = diag_paulis.to_matrices()

            for pauli_matrix, diag_pauli_matrix, factor in zip(paulis_matrices, diag_paulis_matrices, factors):
                transformation_matrix = Operator(circuit).to_matrix()
                transformed_pauli_matrix = transformation_matrix.T.conj() @ pauli_matrix @ transformation_matrix

                assert_unitaries_equivalent(transformed_pauli_matrix, factor * diag_pauli_matrix)
