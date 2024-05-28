import unittest

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary import bit_operations as bitops
from pauliarray.binary import symplectic


class TestSymplecticBitsOperations(unittest.TestCase):
    def test_dot(self):
        zx_strings = np.tri(4, 4, k=0, dtype=np.bool_)

        assert np.all(symplectic.dot(zx_strings, zx_strings) == np.array([0, 0, 2, 4]))

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

        paulis = pa.PauliArray.from_labels(["XXXX", "XXYY", "YYXX", "YYYY"])

        gs_paulis = pa.PauliArray.from_zx_strings(symplectic.gram_schmidt_orthogonalization(paulis.zx_strings))


if __name__ == "__main__":
    unittest.main()
