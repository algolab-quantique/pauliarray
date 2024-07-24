import os
import time
import unittest

import numpy as np

from pauliarray.pauli import operator as op
from pauliarray.pauli import pauli_array as pa
from pauliarray.pauli import weighted_pauli_array as wpa
from pauliarray.pauli.operator import commutator


class TestOperator(unittest.TestCase):
    def test_power(self):
        po1 = op.Operator.from_labels_and_weights(["XY", "IZ"], np.array([1, 2]))

        # Test power with positive integer, simplify = False
        po1_power2 = po1.power(2)
        expected_operator = op.Operator.from_labels_and_weights(["II", "XX"], np.array([5 + 0j, 0 + 0j]))

        self.assertTrue(np.all(po1_power2.weights == expected_operator.weights))
        self.assertTrue(np.all(po1_power2.paulis.zx_strings == expected_operator.paulis.zx_strings))

        # Test power with positive integer, simplify = True
        po1_power2_simplify = po1.power(2, simplify=True)
        expected_operator_simplify = op.Operator.from_labels_and_weights(["II"], np.array([5 + 0j]))

        self.assertTrue(np.all(po1_power2_simplify.weights == expected_operator_simplify.weights))
        self.assertTrue(np.all(po1_power2_simplify.paulis.zx_strings == expected_operator_simplify.paulis.zx_strings))

        # Test power with n = 0
        po1_power0 = po1.power(0)
        identity_operator = op.Operator.identity(po1.num_qubits)
        self.assertEqual(po1_power0.weights, identity_operator.weights)

        # Test power with n = 1
        self.assertEqual(po1.power(1), po1)

        # Test power with n < 0
        self.assertRaises(ValueError, lambda: po1.power(-2))

    def test_compose(self):
        # Test multiply with scalar
        po1 = op.Operator.from_labels_and_weights(["IZ", "IX", "IY", "IZ"], np.array([1, 2, 3, 4]))

        expected_po1_times_two = op.Operator.from_labels_and_weights(["IZ", "IX", "IY", "IZ"], np.array([2, 4, 6, 8]))

        po1_times_two = po1 * 2
        self.assertEqual(po1_times_two, expected_po1_times_two)

        # Test scalar multiplication commutativity
        two_times_po1 = 2 * po1
        self.assertEqual(two_times_po1, expected_po1_times_two)

        # Test multiply po1 with po2
        po2 = op.Operator.from_labels_and_weights(["IX", "IY", "IZ", "IX"], np.array([1, 2, 1, 2]))

        expected_po1_times_po2 = op.Operator.from_labels_and_weights(
            ["II", "IX", "IZ", "IY"], np.array([17, -7j, -5j, 13j])
        )

        self.assertEqual(po1 * po2, expected_po1_times_po2)

        # Test multiply po2 with po1
        expected_po2_times_po1 = op.Operator.from_labels_and_weights(
            ["II", "IX", "IZ", "IY"], np.array([17, 7j, 5j, -13j])
        )

        self.assertEqual(po2 * po1, expected_po2_times_po1)

    # def test_commutator(self):

    def test_eq(self):
        # Test __eq__ method with the same op.Operator
        weights1 = np.array([1, 2, 3, 4])
        labels1 = ["IIIZ", "IIXX", "IYYY", "IIIZ"]
        po = op.Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels1, weights1))

        self.assertEqual(po, po)

        # Test __eq__ on two PauliOperators with the same WeightedPauliArrays but in a different order.
        weights2 = np.array([2, 1, 3, 4])
        labels2 = ["IIXX", "IIIZ", "IYYY", "IIIZ"]
        expected_po = op.Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels2, weights2))

        self.assertEqual(po, expected_po)

    def test_str(self):
        weights = np.array([1, 2, 3, 4])
        labels = ["IIIZ", "IIXX", "IYYY", "IIIZ"]

        po1 = op.Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels, weights))

        self.assertEqual(str(po1), "Operator: num_qubits = 4, num_terms = 4, ...")

    def test_equals(self):
        # Test with the same op.Operator
        po = op.Operator.from_labels_and_weights(["IIIZ", "IIXX", "IYYY", "IIIZ"], np.array([1, 2, 3, 4]))

        self.assertEqual(po, po)

        # Test two PauliOperators with the same WeightedPauliArrays but in a different order.
        other_po = op.Operator.from_labels_and_weights(["IIXX", "IIIZ", "IYYY", "IIIZ"], np.array([2, 1, 3, 4]))

        self.assertEqual(po, other_po)

        # Test two PauliOperators with repeated terms
        other_po = op.Operator.from_labels_and_weights(
            ["IIXX", "IIXX", "IIIZ", "IYYY", "IIIZ"],
            np.array([1, 1, 1, 3, 4]),
        )

        self.assertEqual(po, other_po)

        # Test two PauliOperators of different length
        other_po = op.Operator.from_labels_and_weights(
            ["IIIZ", "IIXX", "IYYY"],
            np.array([1, 2, 3]),
        )

        self.assertNotEqual(po, other_po)

    def test_remove_small_weights(self):
        # Testing op.Operator.remove_small_weights also tests op.Operator.filter_weights()
        weights1 = np.array([1e-15, 2, 3e-15j, 4])
        labels1 = ["IIIZ", "IIXX", "IYYY", "IIIZ"]
        po = op.Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels1, weights1))

        weights2 = np.array([2, 4])
        labels2 = ["IIXX", "IIIZ"]
        expected_po = op.Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels2, weights2))

        po_remove_small_weights = po.remove_small_weights()
        self.assertEqual(po_remove_small_weights, expected_po)

    def test_from_weights_and_labels(self):
        weights = np.array([1, 2, 3, 4])
        labels1 = ["IIIZ", "IIXX", "IYYY", "IIIZ"]
        labels2 = ["IIIX", "IIZX", "IYZY", "IZIZ"]

        po1 = op.Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels1, weights))
        po2 = op.Operator(wpa.WeightedPauliArray.from_labels_and_weights(labels2, weights))

        com_po = commutator(po1, po2)

    def test_clifford_conjugate_pauli_array(self):
        labels = ["IX", "IY", "IZ", "XI", "YI", "ZI", "II", "II"]

        paulis = pa.PauliArray.from_labels(labels).reshape((4, 2))

        po1 = op.Operator.from_labels_and_weights(["II", "XI", "IZ", "XZ"], 0.5 * np.array([1, 1, 1, -1]))
        po2 = op.Operator.from_labels_and_weights(
            ["IX", "IZ"],
            np.sqrt(0.5) * np.array([1, 1]),
        )
        potot = po2.compose_operator(po1)

        reps = 100
        t0 = time.time()
        for i in range(reps):
            new_paulis_1, coefs_1 = potot.clifford_conjugate_pauli_array(paulis)
        print(time.time() - t0)

        t0 = time.time()
        for i in range(reps):
            new_paulis_2, coefs_2 = potot.clifford_conjugate_pauli_array_old(paulis)
        print(time.time() - t0)

        self.assertTrue(np.all(new_paulis_1 == new_paulis_2))
        self.assertTrue(np.all(np.isclose(coefs_1, coefs_2)))

    def test_combine_repeated_terms(self):

        labels = ["XI", "XI", "YI", "ZI", "ZI", "YI", "II", "II"] * 10

        expected_labels = ["XI", "YI", "ZI", "II"]

        po = op.Operator.from_labels_and_weights(labels, np.ones((len(labels))))
        expected_po = op.Operator.from_labels_and_weights(expected_labels, 20 * np.ones((len(expected_labels))))

        new_po = po.combine_repeated_terms()

        self.assertTrue(new_po == expected_po)

    def test_to_npz_from_npz(self):

        po1 = op.Operator.from_labels_and_weights(["XY", "IZ"], np.array([1, 2]))

        po1.to_npz("operator.npz")

        po2 = op.Operator.from_npz("operator.npz")

        self.assertTrue(po1 == po2)

        os.remove("operator.npz")

    def test_to_matrix(self):

        po = 0.5 * op.Operator.from_labels_and_weights(["II", "XI", "IZ", "XZ"], np.array([1, 1, 1, -1]))

        expected_matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        self.assertTrue(np.all(np.isclose(po.to_matrix(sparse=True).toarray(), expected_matrix)))


if __name__ == "__main__":
    unittest.main()
