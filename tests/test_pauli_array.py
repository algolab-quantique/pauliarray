import time
import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.binary import bit_operations as bitops
from pauliarray.utils.pauli_array_library import gen_complete_pauli_array_basis


class TestPauliArray(unittest.TestCase):
    def test_eq(self):
        paulis_1 = pa.PauliArray.from_labels(["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY"])
        paulis_2 = pa.PauliArray.from_labels(["XX", "XI", "XZ", "IX", "YY", "YI", "ZX", "IY"])
        expected_equals = np.array([True, False] * 4)

        equals = paulis_1 == paulis_2

        self.assertTrue(np.all(equals == expected_equals))

    def test_str(self):
        paulis_1 = pa.PauliArray.from_labels(
            [
                [["IIIZ", "IIXX", "IYYY"], ["XXXI", "XXII", "XIII"]],
                [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
            ]
        )

        paulis_2 = pa.PauliArray.from_labels(
            [
                [["IIIZ", "IIXX"], ["XXXI", "XXII"]],
                [["ZIIZ", "ZIXX"], ["ZXXI", "ZXII"]],
            ]
        )

        self.assertEqual(str(paulis_1), "PauliArray: num_qubits = 4, shape = (2, 2, 3), ...")
        self.assertEqual(str(paulis_2), "PauliArray: num_qubits = 4, shape = (2, 2, 2), ...")

    def test_getitem(self):
        labels = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
        paulis_1 = pa.PauliArray.from_labels(labels)

        self.assertTrue(np.all(paulis_1[1:4] == pa.PauliArray.from_labels(labels[1:4])))

    def test_commute_with(self):
        paulis_1 = pa.PauliArray.from_labels(["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"])

        self.assertTrue(np.all((paulis_1[:, None].commute_with(paulis_1[None, :])).diagonal() == True))

    def test_commutator(self):
        paulis = pa.PauliArray.from_labels(
            ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"] * 100
        )

        comm_1, coefs_1 = pa.commutator(paulis[:, None], paulis[None, :])

        comm_2, coefs_2 = pa.commutator2(paulis[:, None], paulis[None, :])

        self.assertTrue(np.all(comm_1 == comm_2))
        self.assertTrue(np.all(np.isclose(coefs_1, coefs_2)))

    def test_bitwisecommute_with(self):
        paulis_1 = pa.PauliArray.from_labels(
            ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
        )

        bw_comms = paulis_1[:, None].bitwise_commute_with(paulis_1[None, :])
        gen_comm = paulis_1[:, None].commute_with(paulis_1[None, :])

        self.assertTrue(np.all(bw_comms == bw_comms * gen_comm))

    def test_generators(self):
        paulis_1 = pa.PauliArray.from_labels(
            ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
        )

        generators = paulis_1.generators()

    def test_generators_with_map(self):
        a_paulis = pa.PauliArray.from_labels(
            ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
        )

        generators, combinaison_map = a_paulis.generators_with_map()

        b_paulis = pa.PauliArray.from_zx_strings(np.dot(combinaison_map, generators.zx_strings))

        self.assertTrue(np.all(a_paulis == b_paulis))

    def test_compose_pauli_array(self):

        all_paulis = gen_complete_pauli_array_basis(1)

        all_prod_paulis, phases = all_paulis[:, None].compose_pauli_array(all_paulis[None, :])

        expected_paulis = pa.PauliArray.from_labels(
            [
                ["I", "Z", "X", "Y"],
                ["Z", "I", "Y", "X"],
                ["X", "Y", "I", "Z"],
                ["Y", "X", "Z", "I"],
            ]
        )
        expected_phases = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1j, -1j],
                [1, -1j, 1, 1j],
                [1, 1j, -1j, 1],
            ]
        )

        self.assertTrue(np.all(all_prod_paulis == expected_paulis))
        self.assertTrue(np.all(phases == expected_phases))

        a_paulis = pa.PauliArray.from_labels(["ZYYY", "XYZY", "YYZI"])
        b_paulis = pa.PauliArray.from_labels(["XXXX", "YYYY", "ZZZZ"])

        expected_paulis = pa.PauliArray.from_labels(
            [
                ["YZZZ", "XIII", "IXXX"],
                ["IZYZ", "ZIXI", "YXIX"],
                ["ZZYX", "IIXY", "XXIZ"],
            ]
        )

        expected_phases = np.array(
            [
                [-1, -1j, -1j],
                [-1j, 1, 1j],
                [-1j, -1j, -1],
            ]
        )

        all_prod_paulis, phases = a_paulis[:, None].compose_pauli_array(b_paulis[None, :])

        self.assertTrue(np.all(all_prod_paulis == expected_paulis))
        self.assertTrue(np.all(phases == expected_phases))

    def test_add_pauli_array(self):

        a_paulis = pa.PauliArray.from_labels(["ZYYY", "XYZY", "YYZI"])
        b_paulis = pa.PauliArray.from_labels(["XXXX", "YYYY", "ZZZZ"])

        operators = a_paulis.add_pauli_array(b_paulis)

        self.assertTrue(np.all(a_paulis.shape == operators.shape))

        a_paulis = pa.PauliArray.from_labels(
            [
                [["IIIZ", "IIXX"], ["XXXI", "XXII"]],
                [["ZIIZ", "ZIXX"], ["ZXXI", "ZXII"]],
            ]
        )
        b_paulis = pa.PauliArray.from_labels(
            [
                [["IXIZ", "IYXX"], ["XXZI", "XXYI"]],
                [["ZIZZ", "XIXX"], ["ZXII", "YXII"]],
            ]
        )

        operators = a_paulis.add_pauli_array(b_paulis)

        self.assertTrue(np.all(a_paulis.shape == operators.shape))

    def test_mul_weights(self):

        paulis = pa.PauliArray.from_labels(["ZYYY", "XYZY", "YYZI"])
        wpaulis_1 = paulis.mul_weights(2)
        wpaulis_2 = paulis.mul_weights([2, 3, 4])

        expected_wpaulis_1 = wpa.WeightedPauliArray.from_labels_and_weights(["ZYYY", "XYZY", "YYZI"], [2, 2, 2])
        expected_wpaulis_2 = wpa.WeightedPauliArray.from_labels_and_weights(["ZYYY", "XYZY", "YYZI"], [2, 3, 4])

        self.assertTrue(np.all(wpaulis_1 == expected_wpaulis_1))
        self.assertTrue(np.all(wpaulis_2 == expected_wpaulis_2))

    def test_to_matrices(self):
        paulis_1 = pa.PauliArray.from_labels(["IX", "XX"])

        matrices = paulis_1.to_matrices()

        expected_matricies = np.array(
            [
                [
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
            ]
        )

        self.assertTrue(np.all(matrices == expected_matricies))

    def test_x(self):
        paulis_basis = gen_complete_pauli_array_basis(1)

        x_op = op.Operator.from_labels_and_weights(["X"], np.array([1]))

        b_paulis, b_factors = paulis_basis.x(0, inplace=False)
        c_paulis, c_factors = x_op.clifford_conjugate_pauli_array_old(paulis_basis)

        self.assertTrue(np.all(np.isclose(b_factors, c_factors)))
        self.assertTrue(np.all(b_paulis == c_paulis))

    def test_s(self):
        paulis_basis = gen_complete_pauli_array_basis(1)

        s_op = op.Operator.from_labels_and_weights(["I", "Z"], np.sqrt(0.5) * np.array([1, -1j]))

        b_paulis, b_factors = paulis_basis.s(0, inplace=False)
        c_paulis, c_factors = s_op.clifford_conjugate_pauli_array_old(paulis_basis)

        self.assertTrue(np.all(np.isclose(b_factors, c_factors)))
        self.assertTrue(np.all(b_paulis == c_paulis))

    def test_h(self):
        paulis_basis = gen_complete_pauli_array_basis(1)

        h_op = op.Operator.from_labels_and_weights(["X", "Z"], np.sqrt(0.5) * np.array([1, 1]))

        b_paulis, b_factors = paulis_basis.h(0, inplace=False)
        c_paulis, c_factors = h_op.clifford_conjugate_pauli_array_old(paulis_basis)

        self.assertTrue(np.all(np.isclose(b_factors, c_factors)))
        self.assertTrue(np.all(b_paulis == c_paulis))

    def test_cx(self):
        paulis_basis = gen_complete_pauli_array_basis(2)

        cx_op = op.Operator.from_labels_and_weights(["II", "IZ", "XI", "XZ"], 0.5 * np.array([1, 1, 1, -1]))

        b_paulis, b_factors = paulis_basis.cx(0, 1, inplace=False)
        c_paulis, c_factors = cx_op.clifford_conjugate_pauli_array_old(paulis_basis)

        self.assertTrue(np.all(np.isclose(b_factors, c_factors)))
        self.assertTrue(np.all(b_paulis == c_paulis))

    def test_cz(self):
        paulis_basis = gen_complete_pauli_array_basis(2)

        cz_op = op.Operator.from_labels_and_weights(["II", "IZ", "ZI", "ZZ"], 0.5 * np.array([1, 1, 1, -1]))

        b_paulis, b_factors = paulis_basis.cz(0, 1, inplace=False)
        c_paulis, c_factors = cz_op.clifford_conjugate_pauli_array_old(paulis_basis)

        self.assertTrue(np.all(np.isclose(b_factors, c_factors)))
        self.assertTrue(np.all(b_paulis == c_paulis))

    def test_clifford_conjugate(self):

        paulis_basis = gen_complete_pauli_array_basis(1)

        h_op = op.Operator.from_labels_and_weights(["X", "Z"], np.sqrt(0.5) * np.array([1, 1]))

        b_paulis, b_factors = paulis_basis.h(0, inplace=False)
        c_paulis, c_factors = paulis_basis.clifford_conjugate(h_op, inplace=False)

        self.assertTrue(np.all(np.isclose(b_factors, c_factors)))
        self.assertTrue(np.all(b_paulis == c_paulis))

    def test_reorder_qubits(self):
        a_paulis = pa.PauliArray.from_labels([["IXYZ"], ["YYYY"]])
        b_paulis = pa.PauliArray.from_labels([["ZYXI"], ["YYYY"]])

        self.assertTrue(np.all(a_paulis.reorder_qubits([3, 2, 1, 0]) == b_paulis))

    def test_from_labels(self):
        paulis_1 = pa.PauliArray.from_labels(["IX", "IY", "IZ", "XI", "YI", "ZI"])

        a_z_bits = np.array([[0, 0], [1, 0], [1, 0], [0, 0], [0, 1], [0, 1]])
        a_x_bits = np.array([[1, 0], [1, 0], [0, 0], [0, 1], [0, 1], [0, 0]])

        self.assertTrue(np.all(a_z_bits == paulis_1.z_strings))
        self.assertTrue(np.all(a_x_bits == paulis_1.x_strings))

    def test_z_string_x_string_to_label(self):
        paulis_1 = pa.PauliArray.from_labels(["ZZXY"])
        label = pa.PauliArray.z_string_x_string_to_label(paulis_1._z_strings[0], paulis_1._x_strings[0])
        expected_label = "ZZXY"

        self.assertEqual(label, expected_label)

    def test_inspect(self):
        # Test inspect() on pa.PauliArray of many elements
        paulis_1 = pa.PauliArray.from_labels(["ZZXY", "XXZY", "XYZI"])
        expected_str_a = "PauliArray\nZZXY\nXXZY\nXYZI"

        self.assertEqual(paulis_1.inspect(), expected_str_a)

        # Test inspect() on pa.PauliArray of one element
        paulis_2 = pa.PauliArray.from_labels("ZZXY")
        expected_str_zzxy = "PauliArray\nZZXY"

        self.assertEqual(paulis_2.inspect(), expected_str_zzxy)

        paulis_3 = pa.PauliArray.from_labels([["IIIZ", "IIXX"], ["XXXI", "XXII"]])
        paulis_3.inspect()

        paulis_4 = pa.PauliArray.from_labels(
            [
                [["IIIZ", "IIXX", "IYYY"], ["XXXI", "XXII", "XIII"]],
                [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
            ]
        )
        paulis_4.inspect()

        paulis_5 = pa.PauliArray.from_labels(
            [
                [
                    [["IIIZ", "IIXX", "IYYY"], ["XXXI", "XXII", "XIII"]],
                    [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
                ],
                [
                    [["IIIZ", "IIXX", "IYYY"], ["XXXI", "XXII", "XIII"]],
                    [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
                ],
            ]
        )
        paulis_5.inspect()

    def test_to_labels(self):
        paulis_1 = pa.PauliArray.from_labels(
            [
                [["IIIZ", "IIXX", "IYYY"], ["XXXI", "XXII", "XIII"]],
                [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
            ]
        )

        labels = paulis_1.to_labels()
        expected_labels = np.array(
            [
                [["IIIZ", "IIXX", "IYYY"], ["XXXI", "XXII", "XIII"]],
                [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
            ]
        )

        self.assertTrue(np.all(labels == expected_labels))

    def test_matrix_from_zx_ints(self):
        paulis_1 = pa.PauliArray.from_labels("ZX")

        z_int = bitops.strings_to_ints(paulis_1.z_strings)[0]
        x_int = bitops.strings_to_ints(paulis_1.x_strings)[0]

        matrix = pa.PauliArray.matrix_from_zx_ints(z_int, x_int, paulis_1.num_qubits)

        expected_matrix = np.array(
            [
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j],
            ]
        )

        self.assertTrue(np.all(matrix == expected_matrix))

    def test_set_item(self):
        paulis_1 = pa.PauliArray.from_labels(
            [
                [["IIIZ", "IIXX", "IYYY"], ["XXXI", "XXII", "XIII"]],
                [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
            ]
        )
        paulis_2 = pa.PauliArray.from_labels([[["IIII", "IIII"], ["IIII", "IIII"]]])

        paulis_1[0, :2, :2] = paulis_2

        expected_a_pauli_array = pa.PauliArray.from_labels(
            [
                [["IIII", "IIII", "IYYY"], ["IIII", "IIII", "XIII"]],
                [["ZIIZ", "ZIXX", "ZYYY"], ["ZXXI", "ZXII", "ZIII"]],
            ]
        )

        self.assertTrue(np.all(paulis_1 == expected_a_pauli_array))

    #

    def test_unique(self):
        paulis_1 = pa.PauliArray.from_labels(
            [
                ["IIIX", "IIIY", "IIIZ"],
                ["IIIX", "IIIY", "IIIZ"],
                ["IIIX", "IIIY", "IIIZ"],
            ]
        )
        paulis_2 = pa.PauliArray.from_labels(
            [
                ["IIIX", "IIIX", "IIIX"],
                ["IIIY", "IIIY", "IIIY"],
                ["IIIZ", "IIIZ", "IIIZ"],
            ]
        )

        unique_a_pauli_array = pa.unique(paulis_1, axis=None)
        unique_b_pauli_array = pa.unique(paulis_2, axis=None)

        self.assertEqual(unique_a_pauli_array.size, 3)
        self.assertEqual(unique_b_pauli_array.size, 3)

    def test_fast_flat_unique(self):

        paulis_1 = pa.PauliArray.from_labels(["IIIX", "IIIY", "IIIZ"] * 10)

        unique_a_pauli_array = pa.fast_flat_unique(paulis_1)

        self.assertEqual(unique_a_pauli_array.size, 3)

    def test_traces(self):
        paulis_1 = pa.PauliArray.from_labels(
            [
                ["II", "IX", "IY", "IZ"],
                ["ZI", "YI", "XI", "II"],
            ]
        )

        expected_traces = np.array([[4, 0, 0, 0], [0, 0, 0, 4]], dtype=int)

        self.assertTrue(np.all(paulis_1.traces() == expected_traces))


class TestPauliArrayFunc(unittest.TestCase):

    def test_argsort(self):
        paulis = pa.PauliArray.from_labels([["XX", "YY", "ZZ"], ["XY", "XX", "XZ"]])

        sorted_args = pa.argsort(paulis)
        expected_sorted_args = [[2, 0, 1], [2, 1, 0]]
        self.assertTrue(np.all(sorted_args == expected_sorted_args))

        sorted_args_axis_0 = pa.argsort(paulis, axis=0)
        expected_sorted_args_axis_0 = [[0, 1, 0], [1, 0, 1]]
        self.assertTrue(np.all(sorted_args_axis_0 == expected_sorted_args_axis_0))

    def test_concatenate(self):
        paulis_1 = pa.PauliArray.from_labels(
            [
                ["IIIX", "IIIY", "IIIZ"],
                ["IIIX", "IIIY", "IIIZ"],
                ["IIIX", "IIIY", "IIIZ"],
            ]
        )
        paulis_2 = pa.PauliArray.from_labels(
            [
                ["IIIX", "IIIX", "IIIX"],
                ["IIIY", "IIIY", "IIIY"],
                ["IIIY", "IIIY", "IIIY"],
            ]
        )

        c_pauli_array = pa.concatenate((paulis_1, paulis_2), 1)

        expected_c_pauli_array = pa.PauliArray.from_labels(
            [
                ["IIIX", "IIIY", "IIIZ", "IIIX", "IIIX", "IIIX"],
                ["IIIX", "IIIY", "IIIZ", "IIIY", "IIIY", "IIIY"],
                ["IIIX", "IIIY", "IIIZ", "IIIY", "IIIY", "IIIY"],
            ]
        )

        self.assertTrue(np.all(c_pauli_array == expected_c_pauli_array))

    def test_commutator(self):
        paulis_1 = pa.PauliArray.from_labels(["IIIX", "IIIY", "IIIZ"])
        paulis_2 = pa.PauliArray.from_labels(["IIIY", "IIIZ", "IIIX"])

        commutator_array, coefs = pa.commutator(paulis_1, paulis_2)

        expected_commutator = pa.PauliArray.from_labels(["IIIZ", "IIIX", "IIIY"])

        expected_coefs = [0.0 + 2.0j, 0.0 + 2.0j, 0.0 + 2.0j]

        self.assertTrue(np.all(commutator_array == expected_commutator))
        self.assertTrue(np.all(coefs == expected_coefs))

    def test_anticommutator(self):
        paulis_1 = pa.PauliArray.from_labels(["IIIX", "IIIY", "IIIZ"])
        paulis_2 = pa.PauliArray.from_labels(["IIIX", "IIIX", "IIIX"])

        anticommutator_array, coefs = pa.anticommutator(paulis_1, paulis_2)

        expected_commutator = pa.PauliArray.from_labels(["IIII", "IIII", "IIII"])

        expected_coefs = [2.0, 0.0, 0.0]

        self.assertTrue(np.all(anticommutator_array == expected_commutator))
        self.assertTrue(np.all(coefs == expected_coefs))

    def test_expand_dims(self):
        pauli_array = pa.PauliArray.from_labels(["IIIX", "IIIY", "IIIZ"])

        expanded_pauli_array = pa.expand_dims(pauli_array, [0, 2])

        expected_pauli_array = pa.PauliArray.from_labels([["IIIX"], ["IIIY"], ["IIIZ"]])

        self.assertTrue(np.all(expanded_pauli_array == expected_pauli_array))

    def test_broadcast_to(self):

        paulis_1 = pa.PauliArray.from_labels(["IIIX", "IIIY", "IIIZ"])
        paulis_2 = pa.PauliArray.from_labels([["IIIX", "IIIY", "IIIZ"]])

        c_pauli_array = pa.broadcast_to(paulis_1, paulis_2.shape)

        self.assertTrue(np.all(paulis_2 == c_pauli_array))

    def test_swapaxes(self):

        paulis = pa.PauliArray.from_labels(
            [
                ["II", "IX", "IY", "IZ"],
                ["ZI", "YI", "XI", "II"],
            ]
        )

        expected_paulis = pa.PauliArray.from_labels(
            [
                ["II", "ZI"],
                ["IX", "YI"],
                ["IY", "XI"],
                ["IZ", "II"],
            ]
        )

        new_paulis = pa.swapaxes(paulis, -1, -2)

        self.assertTrue(np.all(new_paulis == expected_paulis))


if __name__ == "__main__":
    unittest.main()
