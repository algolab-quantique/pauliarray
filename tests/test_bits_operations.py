import unittest

import numpy as np

from pauliarray.binary import bit_operations as bitops


class TestBitsOperations(unittest.TestCase):
    def test_dot(self):
        bits_b = np.tri(4, 4, k=-1, dtype=np.bool_)

        self.assertTrue(np.all(bitops.dot(bits_b, bits_b) == np.arange(4)))

    def test_rank(self):
        bits_b = np.tri(4, 4, k=-1, dtype=np.bool_)

        self.assertEqual(bitops.rank(bits_b), 3)

    def test_kernel(self):

        bits = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.bool_)
        kernel_bits = bitops.kernel(bits)

        self.assertTrue(np.all(bitops.matmul(bits, kernel_bits.T) == 0))

        comp_bits = np.concatenate((bits, kernel_bits), axis=0)
        kernel_bits_2 = bitops.kernel(comp_bits)
        self.assertTrue(kernel_bits_2.shape[0] == 0)

    def test_kernel_2(self):

        bits = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.bool_,
        )

        kernel_bits = bitops.kernel(bits)

        self.assertTrue(np.all(bitops.matmul(bits, kernel_bits.T) == 0))

        comp_bits = np.concatenate((bits, kernel_bits), axis=0)
        kernel_bits_2 = bitops.kernel(comp_bits)

        self.assertTrue(kernel_bits_2.shape[0] == 0)

    def test_row_echelon(self):

        bits = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
            ],
            dtype=np.bool_,
        )

        re_bits = bitops.row_echelon(bits)

        # TODO validate test

    def test_intersection(self):

        bits_1 = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.bool_,
        )
        bits_2 = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.bool_,
        )

        bits_inter = bitops.intersection(bits_1, bits_2)

        rank_1 = bitops.rank(bits_1)
        rank_1p = bitops.rank(np.concatenate((bits_1, bits_inter), axis=0))
        rank_2 = bitops.rank(bits_2)
        rank_2p = bitops.rank(np.concatenate((bits_2, bits_inter), axis=0))

        self.assertEqual(rank_1, rank_1p)
        self.assertEqual(rank_2, rank_2p)

    def test_pack_diagonal(self):

        bits = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.bool_,
        )

        new_bits, row_op, col_order, start_index = bitops.pack_diagonal(bits)

        # print(new_bits.astype(int))

        self.assertTrue(
            np.all(np.mod((row_op.astype(np.uint) @ bits.astype(np.uint)), 2).astype(bool)[:, col_order] == new_bits)
        )


if __name__ == "__main__":
    unittest.main()
