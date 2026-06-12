import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.operator_array_type_1 as opat1
import pauliarray.pauli.operator_array_type_2 as opat2
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.phased_pauli_array as ppa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.transformation import cliffords
from pauliarray.utils.pauli_array_library import gen_complete_pauli_array_basis


class TestCliffordsPauliArray(unittest.TestCase):
    def test_h(self):

        paulis = gen_complete_pauli_array_basis(2)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IX", "ZI", "ZX", "IZ", "IY", "ZZ", "ZY", "XI", "XX", "YI", "YX", "XZ", "XY", "YZ", "YY"]
        )
        # ref_factors = np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1])
        ref_phases = np.array([0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2])

        new_paulis, new_phases = cliffords.h(paulis, [0])

        assert np.all(new_paulis == ref_paulis) and np.all(new_phases == ref_phases), print(paulis, new_paulis)

    def test_s(self):

        paulis = gen_complete_pauli_array_basis(2)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IZ", "ZI", "ZZ", "IY", "IX", "ZY", "ZX", "XI", "XZ", "YI", "YZ", "XY", "XX", "YY", "YX"]
        )
        # ref_factors = np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1])
        ref_phases = np.array([0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2])

        new_paulis, new_phases = cliffords.s(paulis, [0])

        assert np.all(new_paulis == ref_paulis) and np.all(new_phases == ref_phases), print(paulis, new_paulis)

    def test_cx(self):

        paulis = gen_complete_pauli_array_basis(2)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IZ", "ZZ", "ZI", "XX", "XY", "YY", "YX", "XI", "XZ", "YZ", "YI", "IX", "IY", "ZY", "ZX"]
        )
        # ref_factors = np.array([1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1])
        ref_phases = np.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2])

        new_paulis, new_phases = cliffords.cx(paulis, [0], [1])

        assert np.all(new_paulis == ref_paulis) and np.all(new_phases == ref_phases), print(paulis, new_paulis)

    def test_cz(self):

        paulis = gen_complete_pauli_array_basis(2)

        print(paulis.inspect())

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IZ", "ZI", "ZZ", "ZX", "ZY", "IX", "IY", "XZ", "XI", "YZ", "YI", "YY", "YX", "XY", "XX"]
        )
        ref_phases = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0])

        new_paulis, new_phases = cliffords.cz(paulis, [0], [1])

        assert np.all(new_paulis == ref_paulis) and np.all(new_phases == ref_phases), print(paulis, new_paulis)


class TestCliffordsPhasedPauliArray(unittest.TestCase):

    def test_h(self):

        ppaulis = ppa.PhasedPauliArray.from_paulis(gen_complete_pauli_array_basis(2))

        ref_ppaulis = ppa.PhasedPauliArray.from_labels(
            ["II", "IX", "ZI", "ZX", "IZ", "-IY", "ZZ", "-ZY", "XI", "XX", "YI", "YX", "XZ", "-XY", "YZ", "-YY"]
        )

        new_ppaulis = cliffords.h(ppaulis, [0])

        assert np.all(new_ppaulis == ref_ppaulis), print(ppaulis, new_ppaulis)

        ppaulis = ppaulis.add_i_phases(1)

        ref_ppaulis = ppa.PhasedPauliArray.from_labels(
            [
                "iII",
                "iIX",
                "iZI",
                "iZX",
                "iIZ",
                "-iIY",
                "iZZ",
                "-iZY",
                "iXI",
                "iXX",
                "iYI",
                "iYX",
                "iXZ",
                "-iXY",
                "iYZ",
                "-iYY",
            ]
        )

        new_ppaulis = cliffords.h(ppaulis, [0])

        assert np.all(new_ppaulis == ref_ppaulis), print(ppaulis, new_ppaulis)

    def test_s(self):

        ppaulis = ppa.PhasedPauliArray.from_paulis(gen_complete_pauli_array_basis(2))

        ref_ppaulis = ppa.PhasedPauliArray.from_labels(
            ["II", "IZ", "ZI", "ZZ", "IY", "-IX", "ZY", "-ZX", "XI", "XZ", "YI", "YZ", "XY", "-XX", "YY", "-YX"]
        )

        new_ppaulis = cliffords.s(ppaulis, [0])

        assert np.all(new_ppaulis == ref_ppaulis), print(ppaulis, new_ppaulis)

        ppaulis = ppaulis.add_i_phases(1)

        ref_ppaulis = ppa.PhasedPauliArray.from_labels(
            [
                "iII",
                "iIZ",
                "iZI",
                "iZZ",
                "iIY",
                "-iIX",
                "iZY",
                "-iZX",
                "iXI",
                "iXZ",
                "iYI",
                "iYZ",
                "iXY",
                "-iXX",
                "iYY",
                "-iYX",
            ]
        )

        new_ppaulis = cliffords.s(ppaulis, [0])

        assert np.all(new_ppaulis == ref_ppaulis), print(ppaulis, new_ppaulis)

    def test_cx(self):

        ppaulis = ppa.PhasedPauliArray.from_paulis(gen_complete_pauli_array_basis(2))

        ref_ppaulis = ppa.PhasedPauliArray.from_labels(
            ["II", "IZ", "ZZ", "ZI", "XX", "XY", "-YY", "YX", "XI", "XZ", "YZ", "YI", "IX", "IY", "ZY", "-ZX"]
        )

        new_ppaulis = cliffords.cx(ppaulis, [0], [1])

        assert np.all(new_ppaulis == ref_ppaulis), print(ppaulis, new_ppaulis)

    def test_cz(self):

        ppaulis = ppa.PhasedPauliArray.from_paulis(gen_complete_pauli_array_basis(2))

        ref_ppaulis = ppa.PhasedPauliArray.from_labels(
            ["II", "IZ", "ZI", "ZZ", "ZX", "ZY", "IX", "IY", "XZ", "XI", "YZ", "YI", "YY", "-YX", "-XY", "XX"]
        )

        new_ppaulis = cliffords.cz(ppaulis, [0], [1])

        assert np.all(new_ppaulis == ref_ppaulis), print(ppaulis, new_ppaulis)


class TestCliffordsWeightedPauliArray(unittest.TestCase):

    def test_h(self):

        paulis = gen_complete_pauli_array_basis(2)
        wpaulis = wpa.WeightedPauliArray.from_paulis(paulis)

        new_wpaulis = cliffords.h(wpaulis, [0], inplace=True)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IX", "ZI", "ZX", "IZ", "IY", "ZZ", "ZY", "XI", "XX", "YI", "YX", "XZ", "XY", "YZ", "YY"]
        )
        ref_factors = np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1])

        ref_wpaulis = wpa.WeightedPauliArray(ref_paulis, ref_factors)

        assert np.all(new_wpaulis == ref_wpaulis)

    def test_cx(self):

        paulis = gen_complete_pauli_array_basis(2)
        wpaulis = wpa.WeightedPauliArray.from_paulis(paulis)

        new_wpaulis = cliffords.cx(wpaulis, [0], [1], inplace=True)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IZ", "ZZ", "ZI", "XX", "XY", "YY", "YX", "XI", "XZ", "YZ", "YI", "IX", "IY", "ZY", "ZX"]
        )
        ref_factors = np.array([1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1])

        ref_wpaulis = wpa.WeightedPauliArray(ref_paulis, ref_factors)

        assert np.all(new_wpaulis == ref_wpaulis)

    def test_cx_operator(self):

        paulis = gen_complete_pauli_array_basis(2)
        operator = op.Operator.from_paulis(paulis)

        new_operator = cliffords.cx(operator, [0], [1], inplace=True)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IZ", "ZZ", "ZI", "XX", "XY", "YY", "YX", "XI", "XZ", "YZ", "YI", "IX", "IY", "ZY", "ZX"]
        )
        ref_factors = np.array([1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1])

        ref_operator = op.Operator.from_paulis_and_weights(ref_paulis, ref_factors)

        assert np.all(new_operator == ref_operator)

    def test_cx_opat1(self):

        paulis = gen_complete_pauli_array_basis(2)
        operator1 = op.Operator.from_paulis(paulis)
        operator2 = op.Operator.from_paulis(paulis[::-1])

        opa = opat1.OperatorArrayType1.from_operator_list([operator1, operator2])

        new_opa = cliffords.cx(opa, [0], [1], inplace=True)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IZ", "ZZ", "ZI", "XX", "XY", "YY", "YX", "XI", "XZ", "YZ", "YI", "IX", "IY", "ZY", "ZX"]
        )
        ref_factors = np.array([1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1])

        ref_operator1 = op.Operator.from_paulis_and_weights(ref_paulis, ref_factors)
        ref_operator2 = op.Operator.from_paulis_and_weights(ref_paulis[::-1], ref_factors[::-1])

        ref_opa = opat1.OperatorArrayType1.from_operator_list([ref_operator1, ref_operator2])

        assert np.all(new_opa == ref_opa)

    def test_cx_opat2(self):

        paulis = gen_complete_pauli_array_basis(2)
        operator1 = op.Operator.from_paulis(paulis)
        operator2 = op.Operator.from_paulis(paulis[::-1])

        opa = opat2.OperatorArrayType2.from_operator_list([operator1, operator2])

        new_opa = cliffords.cx(opa, [0], [1], inplace=True)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IZ", "ZZ", "ZI", "XX", "XY", "YY", "YX", "XI", "XZ", "YZ", "YI", "IX", "IY", "ZY", "ZX"]
        )
        ref_factors = np.array([1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1])

        ref_operator1 = op.Operator.from_paulis_and_weights(ref_paulis, ref_factors)
        ref_operator2 = op.Operator.from_paulis_and_weights(ref_paulis[::-1], ref_factors[::-1])

        ref_opa = opat2.OperatorArrayType2.from_operator_list([ref_operator1, ref_operator2])

        assert np.all(new_opa == ref_opa)

    def test_h_opat1(self):

        paulis = gen_complete_pauli_array_basis(2)
        operator1 = op.Operator.from_paulis(paulis)
        operator2 = op.Operator.from_paulis(paulis[::-1])

        opa = opat1.OperatorArrayType1.from_operator_list([operator1, operator2])

        new_opa = cliffords.h(opa, [0], inplace=False)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IX", "ZI", "ZX", "IZ", "IY", "ZZ", "ZY", "XI", "XX", "YI", "YX", "XZ", "XY", "YZ", "YY"]
        )
        ref_factors = np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1])

        ref_operator1 = op.Operator.from_paulis_and_weights(ref_paulis, ref_factors)
        ref_operator2 = op.Operator.from_paulis_and_weights(ref_paulis[::-1], ref_factors[::-1])

        ref_opa = opat1.OperatorArrayType1.from_operator_list([ref_operator1, ref_operator2])

        assert np.all(new_opa == ref_opa)

    def test_h_opat2(self):

        paulis = gen_complete_pauli_array_basis(2)
        operator1 = op.Operator.from_paulis(paulis)
        operator2 = op.Operator.from_paulis(paulis[::-1])

        opa = opat2.OperatorArrayType2.from_operator_list([operator1, operator2])

        new_opa = cliffords.h(opa, [0], inplace=True)

        ref_paulis = pa.PauliArray.from_labels(
            ["II", "IX", "ZI", "ZX", "IZ", "IY", "ZZ", "ZY", "XI", "XX", "YI", "YX", "XZ", "XY", "YZ", "YY"]
        )
        ref_factors = np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1])

        ref_operator1 = op.Operator.from_paulis_and_weights(ref_paulis, ref_factors)
        ref_operator2 = op.Operator.from_paulis_and_weights(ref_paulis[::-1], ref_factors[::-1])

        ref_opa = opat2.OperatorArrayType2.from_operator_list([ref_operator1, ref_operator2])

        assert np.all(new_opa == ref_opa)


if __name__ == "__main__":
    unittest.main()
