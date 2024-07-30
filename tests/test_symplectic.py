import unittest

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops
from pauliarray.binary import symplectic


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

        paulis = pa.PauliArray.from_labels(["IIXX", "ZZXI", "IZII", "ZZIX"])
        isotropic_zx_strings = symplectic.isotropic_subspace(paulis.zx_strings)

        self.assertTrue(symplectic.is_isotropic(isotropic_zx_strings))

    def test_gram_schmidt_orthogonalization(self):

        paulis = pa.PauliArray.from_labels(["IIXZ", "IIIX", "IIXI"])

        gs_paulis = pa.PauliArray.from_zx_strings(symplectic.gram_schmidt_orthogonalization(paulis.zx_strings))

    def test_conjugate_subspace(self):

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

        iso_zx_strings = paulis.zx_strings

        lag_zx_strings = symplectic.lagrangian_subspace(iso_zx_strings)
        colag_zx_strings = symplectic.conjugate_subspace(lag_zx_strings)

        sigma_lc = np.mod(symplectic.dot(lag_zx_strings[:, None, :], colag_zx_strings[None, :, :]), 2)
        sigma_ll = np.mod(symplectic.dot(lag_zx_strings[:, None, :], lag_zx_strings[None, :, :]), 2)
        sigma_cc = np.mod(symplectic.dot(colag_zx_strings[:, None, :], colag_zx_strings[None, :, :]), 2)

        self.assertTrue(np.all(sigma_lc == np.eye(paulis.num_qubits)))
        self.assertTrue(np.all(sigma_ll == np.zeros((paulis.num_qubits, paulis.num_qubits))))
        self.assertTrue(np.all(sigma_cc == np.zeros((paulis.num_qubits, paulis.num_qubits))))

    def test_simplify_lagrangian_colagrangian(self):

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

        iso_zx_strings = paulis.zx_strings

        lag_zx_strings = symplectic.lagrangian_subspace(iso_zx_strings)
        colag_zx_strings = symplectic.conjugate_subspace(lag_zx_strings)

        new_lag_zx_strings, new_colag_zx_strings = symplectic.simplify_lagrangian_colagrangian(
            lag_zx_strings, colag_zx_strings
        )

        sigma_lc = symplectic.dot(new_lag_zx_strings[:, None, :], new_colag_zx_strings[None, :, :])
        sigma_ll = np.mod(symplectic.dot(new_lag_zx_strings[:, None, :], new_lag_zx_strings[None, :, :]), 2)
        sigma_cc = np.mod(symplectic.dot(new_colag_zx_strings[:, None, :], new_colag_zx_strings[None, :, :]), 2)

        self.assertTrue(np.all(sigma_lc == np.eye(paulis.num_qubits)))
        self.assertTrue(np.all(sigma_ll == np.zeros((paulis.num_qubits, paulis.num_qubits))))
        self.assertTrue(np.all(sigma_cc == np.zeros((paulis.num_qubits, paulis.num_qubits))))


if __name__ == "__main__":
    unittest.main()
