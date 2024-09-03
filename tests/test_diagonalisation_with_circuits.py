import unittest

import numpy as np
from qiskit import transpile

import pauliarray.pauli.pauli_array as pa
from pauliarray.diagonalisation.commutating_paulis.with_circuits import general_to_diagonal

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


class TestDiagonalisationWithCircuits(unittest.TestCase):

    def test_general_to_diagonal(self):

        for paulis in cases_paulis[0:1]:

            diag_paulis, factors, circuit = general_to_diagonal(paulis, force_trivial_generators=True)

            t_circuit = transpile(circuit)

            print(circuit)

            generators = paulis.generators()

            print(generators.inspect())

            # paulis.cx([0, 2], [1, 3])
            # paulis.h([0, 2])

            print(generators.inspect())
            print(generators.x_strings.astype(int))
            print(generators.z_strings.astype(int))
            # diag_paulis, factors, transformations = general_to_diagonal(paulis)

            # transformed_paulis, transformed_factors = transformations.successive_clifford_conjugate_pauli_array(paulis)

            # self.assertTrue(np.all(diag_paulis == transformed_paulis))
            # self.assertTrue(np.all(np.isclose(factors, transformed_factors)))
            # self.assertTrue(np.all(diag_paulis.is_diagonal()))

            # print(paulis.inspect())
            # print(transformations.inspect())
            # print(diag_paulis.inspect())
