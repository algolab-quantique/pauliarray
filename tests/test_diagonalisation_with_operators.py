import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.operator_array_type_1 as opa
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.diagonalisation.commutating_paulis.with_operators import (
    bitwise_to_diagonal,
    diagonalise_with_operators,
    general_to_bitwise,
    general_to_diagonal,
    single_qubit_cummutating_generators,
)

cases_paulis = [
    pa.PauliArray.from_labels(
        [
            "XX",
            "YY",
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

    def test_general_to_diagonal(self):

        for paulis in cases_paulis:

            (diag_paulis, factors), transformations = general_to_diagonal(paulis)

            transformed_paulis, transformed_factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

            self.assertTrue(np.all(diag_paulis == transformed_paulis))
            self.assertTrue(np.all(np.isclose(factors, transformed_factors)))
            self.assertTrue(np.all(diag_paulis.is_diagonal()))

    def test_diagonalise_with_operators_on_paulis(self):

        for paulis in cases_paulis:

            (diag_paulis, factors), transformations = diagonalise_with_operators(paulis)

            transformed_paulis, transformed_factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

            self.assertTrue(np.all(diag_paulis == transformed_paulis))
            self.assertTrue(np.all(np.isclose(factors, transformed_factors)))
            self.assertTrue(np.all(diag_paulis.is_diagonal()))

    def test_diagonalise_with_operators_on_wpaulis(self):

        for paulis in cases_paulis:

            wpaulis = wpa.WeightedPauliArray(paulis, np.random.random(paulis.shape))

            diag_wpaulis, transformations = diagonalise_with_operators(wpaulis)

            transformed_wpaulis = transformations.successive_clifford_conjugate_pauli_obj(wpaulis)

            self.assertTrue(np.all(diag_wpaulis == transformed_wpaulis))
            self.assertTrue(np.all(diag_wpaulis.is_diagonal()))

    def test_diagonalise_with_operators_on_operator(self):

        for paulis in cases_paulis:

            operator = op.Operator(wpa.WeightedPauliArray(paulis, np.random.random(paulis.shape)))

            diag_operator, transformations = diagonalise_with_operators(operator)

            transformed_operator = transformations.successive_clifford_conjugate_pauli_obj(operator)

            self.assertTrue(np.all(diag_operator == transformed_operator))
            self.assertTrue(np.all(diag_operator.is_diagonal()))

    def test_diagonalise_with_operators_on_operator_array(self):

        for paulis in cases_paulis:

            wpaulis = wpa.WeightedPauliArray(paulis, np.random.random(paulis.shape))

            operators = opa.OperatorArrayType1(wpaulis)

            diag_operators, transformations = diagonalise_with_operators(operators)

            transformed_operators = transformations.successive_clifford_conjugate_pauli_obj(operators)

            self.assertTrue(np.all(diag_operators == transformed_operators))
            self.assertTrue(np.all(diag_operators.is_diagonal()))
