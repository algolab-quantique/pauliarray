import unittest

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_operators import (
    bitwise_to_diagonal,
    general_to_bitwise,
    general_to_diagonal,
    single_qubit_cummutating_generators,
)

cases_paulis = [
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
    pa.PauliArray.from_labels(
        [
            "XZZZZXXZZZZX",
            "YZZZZYXZZZZX",
            "XZZZZXYZZZZY",
            "YZZZZYYZZZZY",
        ]
    ),
]


class TestDiagonalisationWithOperators(unittest.TestCase):

    def test_trivial_cummutating_generators(self):

        for paulis in cases_paulis:
            print(paulis.inspect())
            gen_paulis = single_qubit_cummutating_generators(paulis)

            print(gen_paulis.inspect())

    def test_general_to_diagonal(self):

        for paulis in cases_paulis:

            diag_paulis, factors, transformations = general_to_diagonal(paulis)

            transformed_paulis, transformed_factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

            self.assertTrue(np.all(diag_paulis == transformed_paulis))
            self.assertTrue(np.all(np.isclose(factors, transformed_factors)))
            self.assertTrue(np.all(diag_paulis.is_diagonal()))

            print(paulis.inspect())
            print(transformations.inspect())
            print(diag_paulis.inspect())

    def test_general_to_diagonal_2(self):

        for paulis in cases_paulis:
            diag_paulis, factors, transformations = general_to_diagonal(paulis)

            transformed_paulis, transformed_factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

            self.assertTrue(np.all(diag_paulis == transformed_paulis))
            self.assertTrue(np.all(np.isclose(factors, transformed_factors)))
            self.assertTrue(np.all(diag_paulis.is_diagonal()))
