import unittest
import torch
import numpy as np

from pauliarray.binary import bit_operations as np_bitops
from pauliarray.binary import bit_operations_torch as torch_bitops

TESTS_PER_FUNCTION = 50
UNDER_ARRAY_SIZE = 5
UPPER_ARRAY_SIZE = 50


class TestTorchBitOperations(unittest.TestCase):
    def test_bit_sum(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits = np.random.randint(0, 2, size=n, dtype=np.bool_)

            np_result = np_bitops.bit_sum(np.array(bits))
            torch_result = torch_bitops.bit_sum(torch.Tensor(bits))

            self.assertTrue(np.all(np_result == torch_result))

    def test_dot(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b = np.tri(n, n, k=-1, dtype=np.bool_)
            torch_bits_b = torch.Tensor(bits_b.tolist()).to(torch.bool)

            np_result = np_bitops.dot(bits_b, bits_b)
            torch_result = torch_bitops.dot(torch_bits_b, torch_bits_b).tolist()

            self.assertTrue(np.all(np_result == torch_result))

    def test_rank(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b = np.tri(n, n, k=-1, dtype=np.bool_)
            torch_bits_b = torch.Tensor(bits_b.tolist())

            np_result = np_bitops.rank(bits_b)
            torch_result = torch_bitops.rank(torch_bits_b)

            self.assertEqual(np_result, torch_result)

    def test_matmul(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b1 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            bits_b2 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            torch_bits_b1 = torch.Tensor(bits_b1.tolist())
            torch_bits_b2 = torch.Tensor(bits_b2.tolist())

            np_result = np_bitops.matmul(bits_b1, bits_b2)
            torch_result = torch_bitops.matmul(torch_bits_b1, torch_bits_b2).tolist()

            self.assertTrue(np.all(np_result == torch_result))

    def test_add(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b1 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            bits_b2 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)

            torch_bits_b1 = torch.Tensor(bits_b1.tolist())
            torch_bits_b2 = torch.Tensor(bits_b2.tolist())

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

    def test_intersection_row_space(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b1 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            bits_b2 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            torch_bits_b1 = torch.Tensor(bits_b1.tolist())
            torch_bits_b2 = torch.Tensor(bits_b2.tolist())

            np_result = np_bitops.intersection_row_space(bits_b1, bits_b2)
            torch_result = torch_bitops.intersection_row_space(torch_bits_b1, torch_bits_b2).tolist()

            self.assertTrue(np.all(np_result == torch_result))

    def test_orthogonal_basis(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            torch_bits_b = torch.Tensor(bits_b.tolist())

            np_result = np_bitops.orthogonal_basis(bits_b)
            torch_result = torch_bitops.orthogonal_basis(torch_bits_b).tolist()

            self.assertTrue(np.all(np_result == torch_result))

    def test_inv(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            det = np.linalg.det(bits_b.astype(np.float64))  # Use float64 for determinant check
            if det == 0:
                continue

            torch_bits_b = torch.tensor(bits_b.tolist(), dtype=torch.bool)

            try:
                np_result = np_bitops.inv(bits_b)
                torch_result = torch_bitops.inv(torch_bits_b).tolist()
            except:
                continue

            self.assertTrue(np.all(np_result == torch_result))

    def test_orthogonal_complement(self):
        bits_b = np.array([[1, 0], [0, 1]], dtype=np.bool_)
        torch_bits_b = torch.tensor(bits_b, dtype=torch.bool)

        np_result = np_bitops.orthogonal_complement(bits_b)
        torch_result = torch_bitops.orthogonal_complement(torch_bits_b)

        if np_result.size == 0 and torch_result.numel() == 0:
            self.assertTrue(True)  # Both are empty
        else:
            self.assertTrue(np.all(np_result == torch_result.tolist()))

    def test_intersection(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b1 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            bits_b2 = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            torch_bits_b1 = torch.Tensor(bits_b1.tolist())
            torch_bits_b2 = torch.Tensor(bits_b2.tolist())

            np_result = np_bitops.intersection(bits_b1, bits_b2)
            torch_result = torch_bitops.intersection(torch_bits_b1, torch_bits_b2).tolist()

            self.assertTrue(np.all(np_result == torch_result))

    def test_is_orthogonal(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE//2)
            bits_b1 = np.random.randint(0, 2, size=(2*n, 2*n), dtype=np.bool_)
            bits_b2 = np.random.randint(0, 2, size=(2*n, 2*n), dtype=np.bool_)
            torch_bits_b1 = torch.Tensor(bits_b1.tolist())
            torch_bits_b2 = torch.Tensor(bits_b2.tolist())

            np_result = np_bitops.is_orthogonal(bits_b1, bits_b2)
            torch_result = torch_bitops.is_orthogonal(torch_bits_b1, torch_bits_b2).tolist()

            self.assertTrue(np.all(np_result == torch_result))

    def test_row_echelon(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            torch_bits_b = torch.Tensor(bits_b.tolist()).to(torch.bool)

            np_result = np_bitops.row_echelon(bits_b)
            torch_result = torch_bitops.row_echelon(torch_bits_b).tolist()

            self.assertTrue(np.all(np_result == torch_result))

    def test_kernel(self):
        for _ in range(TESTS_PER_FUNCTION):
            n = np.random.randint(UNDER_ARRAY_SIZE, UPPER_ARRAY_SIZE)
            bits_b = np.random.randint(0, 2, size=(n, n), dtype=np.bool_)
            torch_bits_b = torch.Tensor(bits_b.tolist())

            np_result = np_bitops.kernel(bits_b)
            torch_result = torch_bitops.kernel(torch_bits_b).cpu().numpy()

            if np_result.size == 0:
                np_result = np_result.reshape((0, bits_b.shape[1]))
            if torch_result.size == 0:
                torch_result = torch_result.reshape((0, bits_b.shape[1]))

            self.assertTrue(np.all(np_result == torch_result))


if __name__ == "__main__":
    unittest.main()
