import unittest

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops
from pauliarray.binary import symplectic

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


class TestSymplecticBitsOperations(unittest.TestCase):
    def test_dot(self):
        zx_strings = np.tri(4, 4, k=0, dtype=np.bool_)

        self.assertTrue(np.all(symplectic.dot(zx_strings, zx_strings) == np.array([0, 0, 2, 4])))

    def test_orthogonal_complement(self):

        paulis = pa.PauliArray.from_labels(["IIXX", "ZZXI", "IZII", "ZZIY"])
        subspace_zx_strings = paulis.zx_strings

        orthogonal_zx_strings = symplectic.orthogonal_complement(subspace_zx_strings)

        self.assertTrue(np.all(symplectic.is_orthogonal(subspace_zx_strings[:, None], orthogonal_zx_strings[None, :])))

        self.assertTrue(
            orthogonal_zx_strings.shape[0] == subspace_zx_strings.shape[1] - bitops.rank(subspace_zx_strings)
        )

    def test_isotropic_subspace(self):

        for paulis in cases_paulis:
            isotropic_zx_strings = symplectic.isotropic_subspace(paulis.zx_strings)

            print(pa.PauliArray.from_zx_strings(isotropic_zx_strings).inspect())

            self.assertTrue(symplectic.is_isotropic(isotropic_zx_strings))

    def test_gram_schmidt_orthogonalization(self):

        paulis = pa.PauliArray.from_labels(["IIXZ", "IIIX", "IIXI"])

        gs_paulis = pa.PauliArray.from_zx_strings(symplectic.gram_schmidt_orthogonalization(paulis.zx_strings))

    def test_conjugate_subspace(self):

        for paulis in cases_paulis:

            iso_zx_strings = paulis.zx_strings

            lag_zx_strings = symplectic.lagrangian_subspace(iso_zx_strings)
            colag_zx_strings = symplectic.conjugate_subspace(lag_zx_strings)

            sigma_lc = np.mod(symplectic.dot(lag_zx_strings[:, None, :], colag_zx_strings[None, :, :]), 2)
            sigma_ll = np.mod(symplectic.dot(lag_zx_strings[:, None, :], lag_zx_strings[None, :, :]), 2)
            sigma_cc = np.mod(symplectic.dot(colag_zx_strings[:, None, :], colag_zx_strings[None, :, :]), 2)

            self.assertTrue(np.all(sigma_lc == np.eye(paulis.num_qubits)))
            self.assertTrue(np.all(sigma_ll == np.zeros((paulis.num_qubits, paulis.num_qubits))))
            self.assertTrue(np.all(sigma_cc == np.zeros((paulis.num_qubits, paulis.num_qubits))))

    def test_lagrangian_bitwise_colagrangian_subspaces(self):

        for paulis in cases_paulis:

            iso_zx_strings = paulis.zx_strings

            lag_zx_strings = symplectic.lagrangian_subspace(iso_zx_strings)

            lag_zx_strings, colag_zx_strings = symplectic.lagrangian_bitwise_colagrangian_subspaces(lag_zx_strings)

            sigma_lc = np.mod(symplectic.dot(lag_zx_strings[:, None, :], colag_zx_strings[None, :, :]), 2)
            sigma_ll = np.mod(symplectic.dot(lag_zx_strings[:, None, :], lag_zx_strings[None, :, :]), 2)
            sigma_cc = np.mod(symplectic.dot(colag_zx_strings[:, None, :], colag_zx_strings[None, :, :]), 2)

            self.assertTrue(np.all(sigma_lc == np.eye(paulis.num_qubits)))
            self.assertTrue(np.all(sigma_ll == np.zeros((paulis.num_qubits, paulis.num_qubits))))
            self.assertTrue(np.all(sigma_cc == np.zeros((paulis.num_qubits, paulis.num_qubits))))

            z_strings, x_strings = symplectic.split_zx_strings(colag_zx_strings)
            active_bits = np.logical_or(z_strings, x_strings)

            self.assertTrue(np.all(np.sum(active_bits, axis=1) == 1))

            print(pa.PauliArray.from_zx_strings(lag_zx_strings).inspect())
            print(pa.PauliArray.from_zx_strings(colag_zx_strings).inspect())

            lag_zx_strings_1, colag_zx_strings_1 = symplectic.row_ech_lagrangian_colagrangian(
                lag_zx_strings, colag_zx_strings
            )

            print("Trans 1")
            print(pa.PauliArray.from_zx_strings(lag_zx_strings_1).inspect())
            print(pa.PauliArray.from_zx_strings(colag_zx_strings_1).inspect())

            # lag_zx_strings_2, colag_zx_strings_2 = symplectic.row_ech_qubit_lagrangian_colagrangian(
            #     lag_zx_strings, colag_zx_strings
            # )

            # print("Trans 2")
            # print(pa.PauliArray.from_zx_strings(lag_zx_strings_2).inspect())
            # print(pa.PauliArray.from_zx_strings(colag_zx_strings_2).inspect())

            # colag_zx_strings_3, lag_zx_strings_3 = symplectic.row_ech_lagrangian_colagrangian(
            #     colag_zx_strings, lag_zx_strings
            # )

            # print("Trans 3")
            # print(pa.PauliArray.from_zx_strings(lag_zx_strings_3).inspect())
            # print(pa.PauliArray.from_zx_strings(colag_zx_strings_3).inspect())

            # colag_zx_strings_4, lag_zx_strings_4 = symplectic.row_ech_qubit_lagrangian_colagrangian(
            #     colag_zx_strings, lag_zx_strings
            # )

            # print("Trans 4")
            # print(pa.PauliArray.from_zx_strings(lag_zx_strings_4).inspect())
            # print(pa.PauliArray.from_zx_strings(colag_zx_strings_4).inspect())


if __name__ == "__main__":
    unittest.main()
