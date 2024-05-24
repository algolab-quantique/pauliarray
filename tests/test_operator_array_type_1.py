import time
import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1, commutator, concatenate
from pauliarray.utils.pauli_array_library import gen_complete_pauli_array_basis


class TestOperatorArrayType1(unittest.TestCase):
    def test_init(self):
        # Test error: create OperatorArray with Operators of different num_qubits.
        po1 = op.Operator.from_labels_and_weights(["XX", "ZZ"], np.array([0, 0 + 0j]))
        po2 = op.Operator.from_labels_and_weights(["XX", "YY"], np.array([1 + 1j, 0 + 1j]))
        po3 = op.Operator.from_labels_and_weights(["XXY", "ZZY"], np.array([0j, 1.0e-13j]))

        self.assertEqual(po1.num_qubits, 2)
        self.assertEqual(po2.num_qubits, 2)
        self.assertEqual(po3.num_qubits, 3)

        self.assertRaises(
            ValueError,
            lambda: OperatorArrayType1.from_operator_list([po1, po2, po3]),
        )

        # Test creating a multidimensionnal OperatorArray
        operator_list = [
            [
                op.Operator.from_labels_and_weights(["XX", "ZZ"], np.array([1, 2])),
                op.Operator.from_labels_and_weights(["XX", "YY"], np.array([-1, -2])),
                op.Operator.from_labels_and_weights(["YY"], np.array([4])),
            ],
            [
                op.Operator.from_labels_and_weights(["ZZ", "YY", "XX"], np.array([1, 2, 3])),
                op.Operator.from_labels_and_weights(["XY", "ZZ", "YX"], np.array([1, 2, 3])),
                op.Operator.from_labels_and_weights(["II"], np.array([0])),
            ],
        ]

        operators = OperatorArrayType1.from_operator_list(operator_list)

        self.assertEqual(operators.shape, (2, 3))
        self.assertEqual(operators.num_qubits, 2)

        #

    def test_get_item(self):
        operator_list = [
            [
                op.Operator.from_labels_and_weights(["XX", "ZZ"], np.array([1, 2])),
                op.Operator.from_labels_and_weights(["XX", "YY"], np.array([-1, -2])),
            ],
            [
                op.Operator.from_labels_and_weights(["ZZ", "YY", "XX"], np.array([1, 2, 3])),
                op.Operator.from_labels_and_weights(["XY", "ZZ", "YX"], np.array([1, 2, 3])),
            ],
        ]

        operators = OperatorArrayType1.from_operator_list(operator_list)

        self.assertEqual(operator_list[0][0], operators.get_operator(0, 0))
        self.assertEqual(operator_list[0][1], operators.get_operator(0, 1))
        self.assertEqual(operator_list[1][0], operators.get_operator(1, 0))
        self.assertEqual(operator_list[1][1], operators.get_operator(1, 1))

    def test_commutator(self):
        operator_list_1 = [
            op.Operator.from_labels_and_weights(["XX", "YY", "ZZ"], np.array([1, 2, 3])),
            op.Operator.from_labels_and_weights(["IX", "IZ"], np.array([-1, -2])),
            op.Operator.from_labels_and_weights(["XI", "ZI"], np.array([-1, -2])),
        ]

        operator_list_2 = [
            op.Operator.from_labels_and_weights(["XY", "ZY"], np.array([1, 3])),
            op.Operator.from_labels_and_weights(["YX", "XY"], np.array([2, 4])),
        ]

        operators_1 = OperatorArrayType1.from_operator_list(operator_list_1)
        operators_2 = OperatorArrayType1.from_operator_list(operator_list_2)

        commutators_a = commutator(operators_1[:, None], operators_2[None, :])

        commutators_b_list = list()
        for i, operator_1 in enumerate(operator_list_1):
            commutators_b_list.append(list())
            for j, operator_2 in enumerate(operator_list_2):
                comm1 = op.commutator(operator_1, operator_2).combine_repeated_terms().remove_small_weights()

                commutators_b_list[i].append(comm1)

        commutators_b = np.array(commutators_b_list, dtype=object)
        for i, j in np.ndindex(commutators_a.shape):
            assert commutators_a.get_operator(i, j) == commutators_b[i, j]

    def test_concatenate(self):
        operator_list_1 = [
            op.Operator.from_labels_and_weights(["XX", "YY"], np.array([1, 2])),
            op.Operator.from_labels_and_weights(["IX", "IZ"], np.array([-1, -2])),
            op.Operator.from_labels_and_weights(["ZX", "IZ"], np.array([2, 3])),
        ]

        operator_list_2 = [
            op.Operator.from_labels_and_weights(["XY", "ZY"], np.array([1, 3])),
            op.Operator.from_labels_and_weights(["YX", "XY"], np.array([2, 4])),
        ]

        operators_1 = OperatorArrayType1.from_operator_list(operator_list_1)
        operators_2 = OperatorArrayType1.from_operator_list(operator_list_2)

        operators_3 = concatenate((operators_1, operators_2), axis=0)

        self.assertTrue(np.all(operators_3.shape == (5,)))

    def test_sum(self):

        operators = OperatorArrayType1(wpa.WeightedPauliArray.random((3, 2, 3, 2), 4))

        operators_1 = operators.sum()

        operators_1_exp = op.Operator.empty(operators.num_qubits)
        for idx in np.ndindex(operators.shape):
            operators_1_exp += operators.get_operator(*idx)

        self.assertEqual(operators_1, operators_1_exp)

        operators_2 = operators.sum((1, 2))

        self.assertTrue(np.all(operators_2.shape == np.array([3])))
        self.assertEqual(operators_2.num_terms, 12)

        for i in range(operators.shape[0]):
            operators_2_exp = op.Operator.empty(operators.num_qubits)
            for idx in np.ndindex(operators[i, :, :].shape):
                operators_2_exp += operators.get_operator(i, *idx)
            self.assertEqual(operators_2.get_operator(i), operators_2_exp)

        operators_3 = operators.sum((1, -1))

        self.assertTrue(np.all(operators_3.shape == np.array([3])))
        self.assertEqual(operators_3.num_terms, 12)

        for i in range(operators.shape[0]):
            operators_3_exp = op.Operator.empty(operators.num_qubits)
            for idx in np.ndindex(operators[i, :, :].shape):
                operators_3_exp += operators.get_operator(i, *idx)
            self.assertEqual(operators_3.get_operator(i), operators_3_exp)

    def test_remove_small_weights(self):

        operator_list = [
            op.Operator.from_labels_and_weights(["XX", "YY"], np.array([1, 0])),
            op.Operator.from_labels_and_weights(["IX", "IZ"], np.array([0, -2])),
            op.Operator.from_labels_and_weights(["ZX", "IZ"], np.array([0, 0])),
        ]

        operators = OperatorArrayType1.from_operator_list(operator_list)

        operators = operators.remove_small_weights()

        expected_operator_list = [
            op.Operator.from_labels_and_weights(["XX"], np.array([1])),
            op.Operator.from_labels_and_weights(["IZ"], np.array([-2])),
            op.Operator.from_labels_and_weights(["II"], np.array([0])),
        ]

        expected_operators = OperatorArrayType1.from_operator_list(expected_operator_list)

        self.assertTrue(np.all(operators == expected_operators))

    def test_from_pauli_array(self):

        paulis = pa.PauliArray.random((2, 3, 4), 4)

        operators_1 = OperatorArrayType1.from_pauli_array(paulis)

        self.assertTrue(np.all(operators_1.shape == np.array([2, 3, 4])))

        operators_2 = OperatorArrayType1.from_pauli_array(paulis, summation_axis=(0, 2))

        self.assertTrue(np.all(operators_2.shape == np.array([3])))

        operators_3 = OperatorArrayType1.from_pauli_array(paulis, summation_axis=-1)

        self.assertTrue(np.all(operators_3.shape == np.array([2, 3])))

    def test_x(self):
        paulis = gen_complete_pauli_array_basis(1)
        operators_1 = OperatorArrayType1.from_pauli_array(paulis)

        x_op = op.Operator.from_labels_and_weights(["X"], np.array([1]))

        operators_2 = operators_1.x(0, inplace=False)
        operators_3 = operators_1.clifford_conjugate(x_op)

        self.assertTrue(np.all(operators_2.wpaulis == operators_3.wpaulis))

    def test_mul_weights(self):

        po1 = op.Operator.from_labels_and_weights(["XX", "ZZ"], np.array([1, 2]))
        po2 = op.Operator.from_labels_and_weights(["XX", "YY"], np.array([3, 4]))
        po3 = op.Operator.from_labels_and_weights(["XX", "ZZ"], np.array([5, 6]))

        operators = OperatorArrayType1.from_operator_list([po1, po2, po3])

        new_operators = operators[:, None].mul_weights(np.ones((3, 2)))

        self.assertTrue(np.all(new_operators.shape == (3, 2)))


if __name__ == "__main__":
    unittest.main()
