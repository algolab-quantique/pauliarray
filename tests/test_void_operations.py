import unittest

import numpy as np

from pauliarray.binary import void_operations as vops


class TestVoidsOperations(unittest.TestCase):
    def test_bit_string_to_voids(self):

        num_qubits = 32

        string_shape = (5, 4) + (num_qubits,)
        bit_strings = np.random.choice(np.array([0, 1], dtype=np.uint8), string_shape)

        voids = vops.bit_strings_to_voids(bit_strings)

        re_bit_strings = vops.voids_to_bit_strings(voids, num_qubits)

        self.assertTrue(np.all(bit_strings == re_bit_strings))

    def test_int_bit_sum(self):

        num_qubits = 8

        string_shape = (5, 4) + (num_qubits,)
        strings = np.random.choice(np.array([0, 1], dtype=np.uint8), string_shape)
        ints = np.packbits(strings, axis=-1, bitorder="little")

        res_ints = vops.int_strings_bitcount(ints)
        res_strings = np.sum(strings, axis=-1)

        self.assertTrue(np.all(res_ints == res_strings))

    def test_bitwise_count(self):

        num_qubits = 128

        shape = (5, 4, 3) + (num_qubits,)
        strings = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        voids = vops.bit_strings_to_voids(strings)

        string_sum = np.sum(strings, axis=-1)
        void_sum = vops.bitwise_count(voids)

        self.assertTrue(np.all(string_sum == void_sum))

    def test_bitwise_not(self):

        num_qubits = 16

        shape = (5,) + (num_qubits,)
        strings = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        voids = vops.bit_strings_to_voids(strings)

        voids_not = vops.bitwise_not(voids)
        voids_not_ref = vops.bit_strings_to_voids(1 - strings)

        print(voids_not)
        print(voids_not_ref)

        self.assertTrue(np.all(voids_not_ref == voids_not))

    def test_paded_bitwise_not(self):

        num_qubits = 8

        shape = (5, 4, 2) + (num_qubits,)
        bit_strings = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        voids = vops.bit_strings_to_voids(bit_strings)

        voids_not = vops.paded_bitwise_not(voids, num_qubits)
        voids_not_ref = vops.bit_strings_to_voids(1 - bit_strings)

        self.assertTrue(np.all(voids_not_ref == voids_not))

    def test_bitwise_dot(self):

        num_qubits = 4
        shape = (5,) + (num_qubits,)
        strings_1 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        strings_2 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)

        voids_1 = vops.bit_strings_to_voids(strings_1)
        voids_2 = vops.bit_strings_to_voids(strings_2)

        res_1 = vops.bitwise_dot(voids_1, voids_2)
        res_2 = np.sum(strings_1 * strings_2, axis=-1)

        self.assertTrue(np.all(res_1 == res_2))

    def test_bitwise_and(self):

        num_qubits = 4
        shape = (5,) + (num_qubits,)
        strings_1 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        strings_2 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        strings_3 = strings_1 * strings_2

        voids_1 = vops.bit_strings_to_voids(strings_1)
        voids_2 = vops.bit_strings_to_voids(strings_2)

        voids_3 = vops.bitwise_and(voids_1, voids_2)

        re_strings_3 = vops.voids_to_bit_strings(voids_3, num_qubits)

        self.assertTrue(np.all(re_strings_3 == strings_3))

    def test_bitwise_xor(self):

        num_qubits = 4
        shape = (5,) + (num_qubits,)
        strings_1 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        strings_2 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        strings_3 = np.mod(strings_1 + strings_2, 2)

        voids_1 = vops.bit_strings_to_voids(strings_1)
        voids_2 = vops.bit_strings_to_voids(strings_2)

        voids_3 = vops.bitwise_xor(voids_1, voids_2)

        re_strings_3 = vops.voids_to_bit_strings(voids_3, num_qubits)

        self.assertTrue(np.all(re_strings_3 == strings_3))

    def test_bitwise_or(self):

        num_qubits = 4
        shape = (5,) + (num_qubits,)
        strings_1 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        strings_2 = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        strings_3 = np.logical_or(strings_1, strings_2)

        voids_1 = vops.bit_strings_to_voids(strings_1)
        voids_2 = vops.bit_strings_to_voids(strings_2)

        voids_3 = vops.bitwise_or(voids_1, voids_2)

        re_strings_3 = vops.voids_to_bit_strings(voids_3, num_qubits)

        self.assertTrue(np.all(re_strings_3 == strings_3))

    def test_eq(self):

        num_qubits = 128

        shape = (5, 4, 3) + (num_qubits,)
        strings = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        voids_1 = vops.bit_strings_to_voids(strings)
        voids_2 = vops.bit_strings_to_voids(strings)

        self.assertTrue(np.all(voids_1 == voids_2))

    def test_stich_voids(self):

        num_qubits = 128

        shape = (5, 4, 3) + (num_qubits,)
        strings = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        voids_1 = vops.bit_strings_to_voids(strings)
        voids_2 = vops.bit_strings_to_voids(strings)

        new_voids = vops.stich_voids(voids_1, voids_2)

        self.assertTrue(np.all(new_voids.shape == voids_1.shape))
        self.assertTrue(new_voids.dtype.itemsize == 2 * voids_1.dtype.itemsize)

    def test_split_voids(self):

        num_qubits = 128

        shape = (5, 4, 3) + (num_qubits,)
        strings = np.random.choice(np.array([0, 1], dtype=np.uint8), shape)
        voids_1 = vops.bit_strings_to_voids(strings)
        voids_2 = vops.bit_strings_to_voids(strings)

        new_voids = vops.stich_voids(voids_1, voids_2)

        re_voids_1, re_voids_2 = vops.split_voids(new_voids)

        self.assertTrue(np.all(re_voids_1 == voids_1))
        self.assertTrue(np.all(re_voids_2 == voids_2))
