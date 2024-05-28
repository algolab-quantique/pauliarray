import time
import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
from pauliarray.pauli.operator_array_type_2 import OperatorArrayType2, commutator, concatenate


class TestOperatorArrayType2(unittest.TestCase):
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
            lambda: OperatorArrayType2.from_operator_list([po1, po2, po3]),
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

        operators = OperatorArrayType2.from_operator_list(operator_list)

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

        operators = OperatorArrayType2.from_operator_list(operator_list)

        self.assertEqual(operator_list[0][0], operators.get_operator(0, 0))
        self.assertEqual(operator_list[0][1], operators.get_operator(0, 1))
        self.assertEqual(operator_list[1][0], operators.get_operator(1, 0))
        self.assertEqual(operator_list[1][1], operators.get_operator(1, 1))

    def test_remove_small_weights(self):
        eps = 1e-16
        operator_list = np.array(
            [
                [
                    op.Operator.from_labels_and_weights(["XX", "XX"], np.array([1, eps])),
                    op.Operator.from_labels_and_weights(["XX", "YY"], np.array([-1, eps])),
                ],
                [
                    op.Operator.from_labels_and_weights(["ZZ", "XX", "YY"], np.array([1, eps, 3])),
                    op.Operator.from_labels_and_weights(["XY", "ZZ", "YX"], np.array([1, 2, 3])),
                ],
            ]
        )

        operators = OperatorArrayType2.from_operator_ndarray(operator_list)

        new_operators = operators.remove_small_weights()

        print(new_operators.get_operator(1, 0).inspect())
        print(new_operators.get_operator(1, 1).inspect())

    def test_remove_unused_basis_paulis(self):
        operator_list = np.array(
            [
                [
                    op.Operator.from_labels_and_weights(["ZZ", "ZY"], np.array([1, 0])),
                    op.Operator.from_labels_and_weights(["YY", "ZY"], np.array([-1, 0])),
                ],
                [
                    op.Operator.from_labels_and_weights(["ZZ", "ZY", "YY"], np.array([1, 0, 3])),
                    op.Operator.from_labels_and_weights(["XY", "ZZ", "YX"], np.array([1, 2, 3])),
                ],
            ]
        )

        operators = OperatorArrayType2.from_operator_ndarray(operator_list)
        operators = operators.remove_small_weights()

        operators.remove_unused_basis_paulis()

        print(operators.basis_paulis.inspect())

        # print(operators.get_operator((1, 0)).inspect())

    def test_combine_basis_paulis(self):
        basis_paulis = pa.PauliArray.from_labels(["II", "XX", "II", "XX", "ZZ"])
        weights = np.zeros((3, basis_paulis.size))
        # print
        weights[[0, 0, 0], [0, 1, 2]] = 1
        weights[[1, 1], [1, 3]] = -1
        weights[[2, 2, 2], [1, 3, 4]] = 2

        operators = OperatorArrayType2(basis_paulis, weights)

        print(operators.basis_paulis.inspect())

        print(operators.get_operator(0).inspect())
        operators.combine_basis_paulis()

        print(operators.get_operator(0).inspect())
        # print(operators.get_operator(0).inspect())

    def test_commutator(self):
        operator_list_1 = [
            op.Operator.from_labels_and_weights(["XX", "YY", "ZZ"], np.array([1, 2, 3])),
            op.Operator.from_labels_and_weights(["IX", "IZ"], np.array([-1, -2])),
        ]

        operator_list_2 = [
            op.Operator.from_labels_and_weights(["XY", "ZY"], np.array([1, 3])),
            op.Operator.from_labels_and_weights(["YX", "XY"], np.array([2, 4])),
        ]

        operators_1 = OperatorArrayType2.from_operator_list(operator_list_1)
        operators_2 = OperatorArrayType2.from_operator_list(operator_list_2)

        print(operators_1.shape)
        print(operators_2.shape)

        t0 = time.time()
        commutators_a = commutator(operators_1[:, None], operators_2[None, :])
        print(time.time() - t0)

        t0 = time.time()
        commutators_b_list = list()
        for i, operator_1 in enumerate(operator_list_1):
            commutators_b_list.append(list())
            for j, operator_2 in enumerate(operator_list_2):
                comm1 = op.commutator(operator_1, operator_2).combine_repeated_terms().remove_small_weights()
                # comm2 = commutators_a.get_operator(i, j)

                commutators_b_list[i].append(comm1)

        print(time.time() - t0)

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

        operators_1 = OperatorArrayType2.from_operator_list(operator_list_1)
        operators_2 = OperatorArrayType2.from_operator_list(operator_list_2)

        operators_3 = concatenate((operators_1, operators_2), axis=0)

        for i in range(operators_3.size):
            print(operators_3.get_operator(i).inspect())

        print(operators_3.basis_paulis.inspect())


if __name__ == "__main__":
    unittest.main()
