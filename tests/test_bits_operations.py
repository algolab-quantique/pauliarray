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

    def test_inv(self):

        invertible_matrix = bits = np.array(
            [
                [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        size = bits.shape[0]

        assert bitops.rank(invertible_matrix) == size

        inv_matrix = bitops.inv(invertible_matrix)

        res = np.mod(inv_matrix.astype(int) @ invertible_matrix.astype(int), 2)

        assert np.all(res == np.identity(size, dtype=bool))

    def test_inv_random(self):

        size = 12
        random_mat = np.tril(np.random.choice([0, 1], p=(0.5, 0.5), size=(size, size)).astype(bool))

        invertible_matrix = np.logical_or(random_mat, np.identity(size, dtype=bool))

        order = np.arange(size)
        np.random.shuffle(order)
        invertible_matrix = invertible_matrix[order, :][:, order]

        assert bitops.rank(invertible_matrix) == size

        inv_matrix = bitops.inv(invertible_matrix)

        res = np.mod(inv_matrix.astype(int) @ invertible_matrix.astype(int), 2)

        assert np.all(res == np.identity(size, dtype=bool))

    def test_row_echelon(self):

        bits = np.array(
            [
                [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )

        size = bits.shape[0]

        re_bits = bitops.row_echelon(bits)

        assert np.all(re_bits == np.identity(size, dtype=bool))

    def test_row_echelon_random(self):

        size = 12
        random_mat = np.tril(np.random.choice([0, 1], p=(0.5, 0.5), size=(size, size)).astype(bool))

        bits = np.logical_or(random_mat, np.identity(size, dtype=bool))

        re_bits = bitops.row_echelon(bits)

        assert np.all(re_bits == np.identity(size, dtype=bool))

    def test_row_echelon_with_map(self):

        size = 12
        random_mat = np.tril(np.random.choice([0, 1], p=(0.5, 0.5), size=(size, size)).astype(bool))

        bits = np.logical_or(random_mat, np.identity(size, dtype=bool))

        re_bits, bit_map = bitops.row_echelon_with_map(bits)

        res = np.mod(bits.astype(int) @ bit_map.astype(int), 2)

        assert np.all(res == np.identity(size, dtype=bool))

    def test_row_space_with_map(self):

        bits = np.random.choice([0, 1], p=(0.5, 0.5), size=(12, 15)).astype(bool)

        re_bits, bit_map = bitops.row_space_with_map(bits)

        re_re_bits = bit_map.astype(int) @ re_bits.astype(int)
        mod_re_re_bits = np.mod(re_re_bits, 2)

        assert np.all(mod_re_re_bits == bits)

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


if __name__ == "__main__":
    unittest.main()
