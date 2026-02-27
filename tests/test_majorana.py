import unittest

import numpy as np

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary.matrix_library import build_identity_matrix
from pauliarray.mapping import majorana


class TestJordanWignerMapping(unittest.TestCase):
    def test_majoranas(self):
        mapping_matrix = build_identity_matrix(4)
        real_majoranas, imag_majoranas = majorana.assemble_real_imag_majoranas(mapping_matrix)

        expected_real_majoranas = pa.PauliArray.from_labels(["IIIX", "IIXZ", "IXZZ", "XZZZ"])
        expected_imag_majoranas = pa.PauliArray.from_labels(["IIIY", "IIYZ", "IYZZ", "YZZZ"])

        self.assertTrue(np.all(expected_real_majoranas == real_majoranas))
        self.assertTrue(np.all(expected_imag_majoranas == imag_majoranas))
