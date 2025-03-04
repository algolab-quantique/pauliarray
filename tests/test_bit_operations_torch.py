from pauliarray.binary import bit_operations as num_bit
from pauliarray.binary.bit_operations_torch import *

import unittest
import torch
from torch import Tensor


class TestBitsOperationsTorch(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor_equal(self, t1: Tensor, t2: Tensor) -> bool:
        return torch.equal(t1.cpu(), t2.cpu())

    def test_dot(self):
        # Create lower triangular matrix
        bits_b = torch.tril(torch.ones((4, 4), dtype=torch.bool, device=self.device), diagonal=-1)

        result = dot(bits_b, bits_b)
        expected = torch.tensor([0, 1, 0, 1], device=self.device)

        self.assertTrue(torch.all(result == expected), f"Dot product failed. Got {result}, expected {expected}")

    def test_rank(self):
        bits_b = torch.tril(torch.ones((4, 4), dtype=torch.bool, device=self.device), diagonal=-1)
        self.assertEqual(rank(bits_b), 3, "Rank calculation incorrect")

    def test_kernel(self):
        # First kernel test
        bits = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.bool, device=self.device)
        kernel_bits = kernel(bits)

        if kernel_bits.shape[0] > 0:
            products = matmul(bits, kernel_bits.T)
            self.assertTrue(torch.all(products == False), "Kernel vectors not orthogonal")

        # Second kernel test with concatenated matrix
        comp_bits = torch.cat((bits, kernel_bits), dim=0)
        kernel_bits_2 = kernel(comp_bits)
        self.assertEqual(kernel_bits_2.shape[0], 0, "Kernel should be empty for full rank matrix")

    def test_kernel_2(self):
        # Complex kernel test
        bits = torch.tensor([
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
        ], dtype=torch.bool, device=self.device)

        kernel_bits = kernel(bits)
        products = matmul(bits, kernel_bits.T)
        self.assertTrue(torch.all(products == False), "Kernel vectors not orthogonal")

        comp_bits = torch.cat((bits, kernel_bits), dim=0)
        kernel_bits_2 = kernel(comp_bits)
        self.assertEqual(kernel_bits_2.shape[0], 0, "Kernel should be empty for full rank matrix")

    def test_row_echelon(self):
        bits = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.bool, device=self.device)

        re_bits = row_echelon(bits)

        lead_col = -1
        for i, row in enumerate(re_bits):
            non_zero = torch.where(row)[0]
            if len(non_zero) > 0:
                new_lead = non_zero[0].item()  # Convert to scalar
                self.assertGreaterEqual(new_lead, lead_col, "Row echelon form violation")
                lead_col = new_lead

    def test_intersection(self):
        bits_1 = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ], dtype=torch.bool, device=self.device)

        bits_2 = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=torch.bool, device=self.device)

        bits_inter = intersection(bits_1, bits_2)

        # Test rank preservation
        rank_1 = rank(bits_1)
        rank_1p = rank(torch.cat((bits_1, bits_inter), dim=0))
        self.assertEqual(rank_1, rank_1p, "Rank 1 changed after intersection concatenation")

        rank_2 = rank(bits_2)
        rank_2p = rank(torch.cat((bits_2, bits_inter), dim=0))
        self.assertEqual(rank_2, rank_2p, "Rank 2 changed after intersection concatenation")

    def test_inverse(self):
        # Test invertible matrix
        matrix = torch.tensor([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=torch.bool, device=self.device)

        try:
            inv_matrix = inv(matrix)
            product = matmul(matrix, inv_matrix)
            identity = torch.eye(3, dtype=torch.bool, device=self.device)
            self.assertTrue(torch.all(product == identity), "Inverse verification failed")
        except ValueError:
            self.fail("Valid inverse should not raise exception")

        # Test singular matrix
        singular = torch.ones((2, 2), dtype=torch.bool, device=self.device)
        with self.assertRaises(ValueError):
            inv(singular)


if __name__ == "__main__":
    unittest.main()
