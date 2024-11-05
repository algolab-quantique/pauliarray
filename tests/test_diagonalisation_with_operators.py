import unittest

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_operators import (
    bitwise_to_diagonal,
    general_to_bitwise,
    general_to_diagonal,
)


class TestDiagonalisationWithOperators(unittest.TestCase):
    def test_general_to_diagonal(self):

        paulis = pa.PauliArray.from_labels(["XXXX", "XXYY", "YYXX", "YYYY"])

        diag_paulis, factors, transformations = general_to_diagonal(paulis)

        transformed_paulis, transformed_factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

        self.assertTrue(np.all(diag_paulis == transformed_paulis))
        self.assertTrue(np.all(np.isclose(factors, transformed_factors)))
        self.assertTrue(np.all(diag_paulis.is_diagonal()))

    def test_general_to_diagonal_2(self):

        paulis = pa.PauliArray.from_labels(
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
        )

        diag_paulis, factors, transformations = general_to_diagonal(paulis)

        transformed_paulis, transformed_factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

        self.assertTrue(np.all(diag_paulis == transformed_paulis))
        self.assertTrue(np.all(np.isclose(factors, transformed_factors)))
        self.assertTrue(np.all(diag_paulis.is_diagonal()))
