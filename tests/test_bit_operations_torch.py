from pauliarray.binary import bit_operations as np_bitops
from pauliarray.binary import bit_operations_torch as torch_bitops
import unittest
import torch
import numpy as np


class TestTorchBitOperations(unittest.TestCase):
    def test_bit_sum(self):
        bits = [0, 1, 0, 1, 1, 0]

        np_result = np_bitops.bit_sum(np.array(bits))
        torch_result = torch_bitops.bit_sum(torch.Tensor(bits))

        self.assertTrue(np.all(np_result == torch_result))

    def test_dot(self):
        bits_b = np.tri(4, 4, k=-1, dtype=np.bool_)
        torch_bits_b = torch.Tensor(bits_b.tolist()).to(torch.bool)

        np_result = np_bitops.dot(bits_b, bits_b)
        torch_result = torch_bitops.dot(torch_bits_b, torch_bits_b).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_rank(self):
        bits_b = np.tri(4, 4, k=-1, dtype=np.bool_)
        torch_bits_b = torch.Tensor(bits_b.tolist())

        np_result = np_bitops.rank(bits_b)
        torch_result = torch_bitops.rank(torch_bits_b)

        self.assertEqual(np_result, torch_result)

    def test_kernel(self):
        bits = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.bool_)
        torch_bits = torch.Tensor(bits.tolist())

        np_kernel = np_bitops.kernel(bits)
        torch_kernel = torch_bitops.kernel(torch_bits)

        np_result = np_bitops.matmul(bits, np_kernel.T)
        torch_result = torch_bitops.matmul(torch_bits, torch_kernel.T).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_matmul(self):
        m1 = [[1, 0], [0, 1]]
        m2 = [[0, 1], [1, 0]]
        bits_b1 = np.array(m1, dtype=np.bool_)
        bits_b2 = np.array(m2, dtype=np.bool_)
        torch_bits_b1 = torch.Tensor(m1)
        torch_bits_b2 = torch.Tensor(m2)

        np_result = np_bitops.matmul(bits_b1, bits_b2)
        torch_result = torch_bitops.matmul(torch_bits_b1, torch_bits_b2).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_add(self):
        m1 = [[1, 0], [0, 1]]
        m2 = [[0, 1], [1, 0]]
        bits_b1 = np.array(m1, dtype=np.bool_)
        bits_b2 = np.array(m2, dtype=np.bool_)
        torch_bits_b1 = torch.Tensor(m1)
        torch_bits_b2 = torch.Tensor(m2)

        np_result = np_bitops.add(bits_b1, bits_b2)
        torch_result = torch_bitops.add(torch_bits_b1, torch_bits_b2).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_strings_to_ints(self):
        m = [[1, 0], [0, 1]]
        bits_b = np.array(m, dtype=np.bool_)
        torch_bits_b = torch.Tensor(m)

        np_result = np_bitops.strings_to_ints(bits_b)
        torch_result = torch_bitops.strings_to_ints(torch_bits_b).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_row_echelon(self):
        m = [[1, 0], [0, 1]]
        bits_b = np.array(m, dtype=np.bool_)
        torch_bits_b = torch.Tensor(m)

        np_result = np_bitops.row_echelon(bits_b)
        torch_result = torch_bitops.row_echelon(torch_bits_b).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_intersection_row_space(self):
        bits_b1 = np.array([[1, 0], [0, 1]], dtype=np.bool_)
        bits_b2 = np.array([[0, 1], [1, 0]], dtype=np.bool_)
        torch_bits_b1 = torch.Tensor(bits_b1.tolist())
        torch_bits_b2 = torch.Tensor(bits_b2.tolist())

        np_result = np_bitops.intersection_row_space(bits_b1, bits_b2)
        torch_result = torch_bitops.intersection_row_space(torch_bits_b1, torch_bits_b2).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_orthogonal_basis(self):
        bits_b = np.array([[1, 0], [0, 1]], dtype=np.bool_)
        torch_bits_b = torch.Tensor(bits_b.tolist())

        np_result = np_bitops.orthogonal_basis(bits_b)
        torch_result = torch_bitops.orthogonal_basis(torch_bits_b).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_inv(self):
        bits_b = np.array([[1, 0], [0, 1]], dtype=np.bool_)
        torch_bits_b = torch.Tensor(bits_b.tolist())

        np_result = np_bitops.inv(bits_b)
        torch_result = torch_bitops.inv(torch_bits_b).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_orthogonal_basis(self):
        bits_b = np.array([[1, 0], [0, 1]], dtype=np.bool_)
        torch_bits_b = torch.Tensor(bits_b.tolist())

        np_result = np_bitops.orthogonal_basis(bits_b)
        torch_result = torch_bitops.orthogonal_basis(torch_bits_b).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_orthogonal_complement(self):
        bits_b = np.array([[1, 0], [0, 1]], dtype=np.bool_)
        torch_bits_b = torch.Tensor(bits_b.tolist())

        np_result = np_bitops.orthogonal_complement(bits_b)
        torch_result = torch_bitops.orthogonal_complement(torch_bits_b).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_intersection(self):
        bits_b1 = np.array([[1, 0], [0, 1]], dtype=np.bool_)
        bits_b2 = np.array([[0, 1], [1, 0]], dtype=np.bool_)
        torch_bits_b1 = torch.Tensor(bits_b1.tolist())
        torch_bits_b2 = torch.Tensor(bits_b2.tolist())

        np_result = np_bitops.intersection(bits_b1, bits_b2)
        torch_result = torch_bitops.intersection(torch_bits_b1, torch_bits_b2).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_is_orthogonal(self):
        m1 = [[1, 0], [0, 1]]
        m2 = [[0, 1], [1, 0]]
        bits_b1 = np.array(m1, dtype=np.bool_)
        bits_b2 = np.array(m2, dtype=np.bool_)
        torch_bits_b1 = torch.Tensor(bits_b1)
        torch_bits_b2 = torch.Tensor(bits_b2)

        np_result = np_bitops.is_orthogonal(bits_b1, bits_b2)
        torch_result = torch_bitops.is_orthogonal(torch_bits_b1, torch_bits_b2).tolist()

        self.assertTrue(np.all(np_result == torch_result))

    def test_row_echelon_random(self):
        np.random.seed(0)
        torch.manual_seed(0)

        for _ in range(10):
            bits_b = np.random.randint(0, 2, size=(5, 5), dtype=np.bool_)
            torch_bits_b = torch.Tensor(bits_b.tolist()).to(torch.bool)

            np_result = np_bitops.row_echelon(bits_b)
            torch_result = torch_bitops.row_echelon(torch_bits_b).tolist()
            print("NumPy result:")
            print(np_result.astype(int))
            print("PyTorch result:")
            print([[int(x) for x in row] for row in torch_result])
            self.assertTrue(np.all(np_result == torch_result))

    def test_kernel_random(self):
        np.random.seed(0)
        torch.manual_seed(0)

        for _ in range(10):
            bits_b = np.random.randint(0, 2, size=(5, 5), dtype=np.bool_)
            torch_bits_b = torch.Tensor(bits_b.tolist())

            np_result = np_bitops.kernel(bits_b)
            torch_result = torch_bitops.kernel(torch_bits_b).tolist()

            self.assertTrue(np.all(np_result == torch_result))


if __name__ == "__main__":
    unittest.main()
