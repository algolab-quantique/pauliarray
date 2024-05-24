import os
import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.utils.pauli_array_library import gen_complete_pauli_array_basis


class TestWeightedPauliArray(unittest.TestCase):
    def test_str(self):
        weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        wpaulis_1 = wpa.WeightedPauliArray.from_labels_and_weights(
            [["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"]], weights
        )

        self.assertEqual(str(wpaulis_1), "WeightedPauliArray: num_qubits = 4, shape = (3, 3), ...")

    def test_from_weights_and_labels(self):
        weights = np.array([1, 2, 3, 4])
        labels = ["IIIZ", "IIXX", "IYYY", "IIIZ"]

        wpaulis = wpa.WeightedPauliArray.from_labels_and_weights(labels, weights)

    def test_extract(self):
        weights1 = np.array([1, 2, 3, 4])
        labels1 = ["IIIZ", "IIXX", "IYYY", "IIIZ"]
        wpaulis = wpa.WeightedPauliArray.from_labels_and_weights(labels1, weights1)

        weights2 = np.array([1, 4])
        labels2 = ["IIIZ", "IIIZ"]
        expected_wpaulis = wpa.WeightedPauliArray.from_labels_and_weights(labels2, weights2)

        extracted_wpaulis = wpaulis.extract([True, False, False, True])

        self.assertTrue(np.all(extracted_wpaulis == expected_wpaulis))

        self.assertRaises(ValueError, lambda: wpaulis.extract([True, False, False, True, True]))

    def test_x(self):
        paulis = gen_complete_pauli_array_basis(1)
        wpaulis_1 = wpa.WeightedPauliArray.from_paulis(paulis)

        x_op = op.Operator.from_labels_and_weights(["X"], np.array([1]))

        wpaulis_2 = wpaulis_1.x(0, inplace=False)
        wpaulis_3 = wpaulis_1.clifford_conjugate(x_op)

        self.assertTrue(np.all(wpaulis_2 == wpaulis_3))

    def test_s(self):
        paulis = gen_complete_pauli_array_basis(1)
        wpaulis_1 = wpa.WeightedPauliArray.from_paulis(paulis)

        s_op = op.Operator.from_labels_and_weights(["I", "Z"], np.sqrt(0.5) * np.array([1, -1j]))

        wpaulis_2 = wpaulis_1.s(0, inplace=False)
        wpaulis_3 = wpaulis_1.clifford_conjugate(s_op)

        self.assertTrue(np.all(wpaulis_2 == wpaulis_3))

    def test_h(self):
        paulis = gen_complete_pauli_array_basis(1)
        wpaulis_1 = wpa.WeightedPauliArray.from_paulis(paulis)

        h_op = op.Operator.from_labels_and_weights(["X", "Z"], np.sqrt(0.5) * np.array([1, 1]))

        wpaulis_2 = wpaulis_1.h(0, inplace=False)
        wpaulis_3 = wpaulis_1.clifford_conjugate(h_op)

        self.assertTrue(np.all(wpaulis_2 == wpaulis_3))

    def test_cx(self):
        paulis = gen_complete_pauli_array_basis(2)
        wpaulis_1 = wpa.WeightedPauliArray.from_paulis(paulis)

        cx_op = op.Operator.from_labels_and_weights(["II", "IZ", "XI", "XZ"], 0.5 * np.array([1, 1, 1, -1]))

        wpaulis_2 = wpaulis_1.cx(0, 1, inplace=False)
        wpaulis_3 = wpaulis_1.clifford_conjugate(cx_op)

        self.assertTrue(np.all(wpaulis_2 == wpaulis_3))

    def test_cz(self):
        paulis = gen_complete_pauli_array_basis(2)
        wpaulis_1 = wpa.WeightedPauliArray.from_paulis(paulis)

        cz_op = op.Operator.from_labels_and_weights(["II", "IZ", "ZI", "ZZ"], 0.5 * np.array([1, 1, 1, -1]))

        wpaulis_2 = wpaulis_1.cz(0, 1, inplace=False)
        wpaulis_3 = wpaulis_1.clifford_conjugate(cz_op)

        self.assertTrue(np.all(wpaulis_2 == wpaulis_3))

    def test_to_npz_from_npz(self):

        weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        wpaulis1 = wpa.WeightedPauliArray.from_labels_and_weights(
            [["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"]], weights
        )

        wpaulis1.to_npz("wpaulis.npz")

        wpaulis2 = wpa.WeightedPauliArray.from_npz("wpaulis.npz")

        self.assertTrue(np.all(wpaulis1 == wpaulis2))

        os.remove("wpaulis.npz")


class TestWeightedPauliArrayFunc(unittest.TestCase):
    def test_concatenate(self):
        weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        wpaulis_1 = wpa.WeightedPauliArray.from_labels_and_weights(
            [["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"]], weights
        )
        wpaulis_2 = wpa.WeightedPauliArray.from_labels_and_weights(
            [["IIIX", "IIIX", "IIIX"], ["IIIY", "IIIY", "IIIY"], ["IIIY", "IIIY", "IIIY"]], weights
        )

        wpaulis_3 = wpa.concatenate((wpaulis_1, wpaulis_2), 1)

    def test_commutator(self):
        weights = np.array([1, 2, 3])

        wpaulis_1 = wpa.WeightedPauliArray.from_labels_and_weights(["IIIX", "IIIY", "IIIZ"], weights)
        wpaulis_2 = wpa.WeightedPauliArray.from_labels_and_weights(["IIIY", "IIIZ", "IIIZ"], weights)

        commutators = wpa.commutator(wpaulis_1, wpaulis_2)

    def test_anticommutator(self):
        weights = np.array([1, 2, 3])

        wpaulis_1 = wpa.WeightedPauliArray.from_labels_and_weights(["IIIX", "IIIY", "IIIZ"], weights)
        wpaulis_2 = wpa.WeightedPauliArray.from_labels_and_weights(["IIIY", "IIIZ", "IIIZ"], weights)

        anticommutators = wpa.anticommutator(wpaulis_1, wpaulis_2)


if __name__ == "__main__":
    unittest.main()
