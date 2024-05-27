import os
import time
import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.operator_array_type_1 as opa
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.mapping.fermion import JordanWigner


class TestJordanWignerMapping(unittest.TestCase):
    def test_majoranas(self):
        mapping = JordanWigner(4)
        real_majoranas, imag_majoranas = mapping.majoranas()

        expected_real_majoranas = pa.PauliArray.from_labels(["IIIX", "IIXZ", "IXZZ", "XZZZ"])
        expected_imag_majoranas = pa.PauliArray.from_labels(["IIIY", "IIYZ", "IYZZ", "YZZZ"])

        self.assertTrue(np.all(expected_real_majoranas == real_majoranas))
        self.assertTrue(np.all(expected_imag_majoranas == imag_majoranas))

    def test_creation_annihilation_operators(self):
        mapping = JordanWigner(4)
        creation_operators, annihilation_operators = mapping.assemble_creation_annihilation_operators()

        antis = opa.anticommutator(
            creation_operators[:, None], creation_operators[None, :], combine_repeated_terms=True
        )

        self.assertTrue(np.all(antis.weights == 0))

        antis = opa.anticommutator(
            annihilation_operators[:, None], annihilation_operators[None, :], combine_repeated_terms=True
        )

        self.assertTrue(np.all(antis.weights == 0))

        antis = opa.anticommutator(
            creation_operators[:, None], annihilation_operators[None, :], combine_repeated_terms=True
        )

        self.assertTrue(np.all(antis.weights[:, :, 0] == np.eye(4)))

    def test_occupation_operators(self):
        mapping = JordanWigner(4)

        occupation_operators = mapping.occupation_operators()

        creation_operators, annihilation_operators = mapping.assemble_creation_annihilation_operators()
        expected_occupation_operators = creation_operators.compose_operator_array_type_1(annihilation_operators)

        for i in range(expected_occupation_operators.size):
            self.assertTrue(occupation_operators.get_operator(i) == expected_occupation_operators.get_operator(i))

    def test_one_body_to_pauli_operator_from_tensor(self):

        # filename = "h2_mo_integrals_d_0750.npz"
        filename = "lih_mo_integrals_d_1600.npz"

        file_path = os.path.join("tests/data/integrals", filename)
        npzfile = np.load(file_path)
        one_body = npzfile["one_body"]

        mapping = JordanWigner(one_body.shape[0])

        one_body_operator = mapping.one_body_operator_from_array(one_body).simplify()
        expected_one_body_operator = mapping.assemble_one_body_operator_array().mul_weights(one_body).sum().simplify()

        self.assertTrue(one_body_operator == expected_one_body_operator)

    def test_one_body_to_pauli_operator_from_sparse(self):

        # filename = "h2_mo_integrals_d_0750.npz"
        filename = "lih_mo_integrals_d_1600.npz"

        file_path = os.path.join("tests/data/integrals", filename)
        npzfile = np.load(file_path)
        one_body = npzfile["one_body"]

        mapping = JordanWigner(one_body.shape[0])

        locations = np.where(one_body)
        values = one_body[locations]

        one_body_operator = mapping.one_body_operator_from_sparse(locations, values).simplify()
        expected_one_body_operator = mapping.assemble_one_body_operator_array().mul_weights(one_body).sum().simplify()

        self.assertTrue(one_body_operator == expected_one_body_operator)

    def test_two_body_to_pauli_operator_from_tensor(self):

        # filename = "h2_mo_integrals_d_0750.npz"
        filename = "lih_mo_integrals_d_1600.npz"
        # filename = "nh3_mo_integrals_costum.npz"

        file_path = os.path.join("tests/data/integrals", filename)
        npzfile = np.load(file_path)
        two_body = npzfile["two_body"]

        mapping = JordanWigner(two_body.shape[0])

        two_body_operator = mapping.two_body_operator_from_array(two_body).simplify()
        expected_two_body_operator = mapping.assemble_two_body_operator_array().mul_weights(two_body).sum().simplify()

        self.assertTrue(two_body_operator == expected_two_body_operator)

    def test_two_body_to_pauli_operator_from_sparse(self):

        # filename = "h2_mo_integrals_d_0750.npz"
        # filename = "lih_mo_integrals_d_1600.npz"
        filename = "nh3_mo_integrals_costum.npz"

        # filename = "c2h4_mo_integrals.npz"

        file_path = os.path.join("tests/data/integrals", filename)
        npzfile = np.load(file_path)
        two_body = npzfile["two_body"]

        mapping = JordanWigner(two_body.shape[0])

        locations = np.where(np.abs(two_body) > 1e-12)
        values = two_body[locations]

        two_body_operator = mapping.two_body_operator_from_sparse(locations, values).simplify()
        expected_two_body_operator = mapping.assemble_two_body_operator_array().mul_weights(two_body).sum().simplify()

        self.assertTrue(two_body_operator == expected_two_body_operator)

    def test_assemble_qubit_hamiltonian_from_sparses(self):

        # filename = "h2_mo_integrals_d_0750.npz"
        # filename = "lih_mo_integrals_d_1600.npz"
        filename = "nh3_mo_integrals_costum.npz"

        # filename = "c2h4_mo_integrals.npz"

        file_path = os.path.join("tests/data/integrals", filename)

        npzfile = np.load(file_path)

        one_body = npzfile["one_body"]
        two_body = npzfile["two_body"]

        mapping = JordanWigner(two_body.shape[0])

        one_body_locations = np.where(np.abs(one_body) > 1e-12)
        one_body_values = one_body[one_body_locations]

        two_body_locations = np.where(np.abs(two_body) > 1e-12)
        two_body_values = two_body[two_body_locations]

        hamiltonian = mapping.assemble_qubit_hamiltonian_from_sparses(
            (one_body_locations, one_body_values), (two_body_locations, two_body_values)
        )

        expected_one_body_operator = mapping.assemble_one_body_operator_array().mul_weights(one_body).sum()
        expected_two_body_operator = mapping.assemble_two_body_operator_array().mul_weights(two_body).sum()

        expected_hamiltonian = (expected_one_body_operator + expected_two_body_operator).simplify()

        self.assertTrue(hamiltonian == expected_hamiltonian)


if __name__ == "__main__":
    unittest.main()
