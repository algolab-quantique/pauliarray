import os
import unittest

import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.phased_pauli_array as ppa
from pauliarray.utils.pauli_array_library import gen_complete_pauli_array_basis


class TestPhasedPauliArray(unittest.TestCase):
    def test_split_phases_pauli_labels(self):

        labels = [
            ["IZ", "1IZ"],
            ["+IZ", "+1IZ"],
            ["-IZ", "-1IZ"],
            ["iIZ", "-iIZ"],
            ["+iIZ", "+1iIZ"],
            ["-iIZ", "-1iIZ"],
            ["+jIZ", "+1jIZ"],
            ["-jIZ", "-1jIZ"],
        ]

        ref_phases = np.array([[0, 0], [0, 0], [2, 2], [3, 1], [3, 3], [1, 1], [3, 3], [1, 1]])

        ref_pauli_labels = [
            ["IZ", "IZ"],
            ["IZ", "IZ"],
            ["IZ", "IZ"],
            ["IZ", "IZ"],
            ["IZ", "IZ"],
            ["IZ", "IZ"],
            ["IZ", "IZ"],
            ["IZ", "IZ"],
        ]

        phases, pauli_labels = ppa.PhasedPauliArray.split_phases_pauli_labels(labels)

        assert np.all(phases == ref_phases)
        assert np.all(pauli_labels == ref_pauli_labels)

    def test_from_labels(self):

        ppaulis = ppa.PhasedPauliArray.from_labels(
            [["IIIX", "+IIIY", "-IIIZ"], ["iIIIX", "+iIIIY", "-iIIIZ"], ["jIIIX", "+jIIIY", "-jIIIZ"]]
        )

        ref_paulis = pa.PauliArray.from_labels(
            [["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"]]
        )
        ref_phases = np.array([[0, 0, 2], [3, 3, 1], [3, 3, 1]])

        assert np.all(ppaulis.paulis == ref_paulis)
        assert np.all(ppaulis.phases == ref_phases)

    def test_compose_phased_pauli_array(self):

        all_ppaulis = ppa.PhasedPauliArray.from_paulis(gen_complete_pauli_array_basis(1))

        all_prod_ppaulis = all_ppaulis[:, None].compose_phased_pauli_array(all_ppaulis[None, :])

        expected_ppaulis = ppa.PhasedPauliArray.from_labels(
            [
                ["I", "Z", "X", "Y"],
                ["Z", "I", "iY", "-iX"],
                ["X", "-iY", "I", "iZ"],
                ["Y", "iX", "-iZ", "I"],
            ]
        )

        self.assertTrue(np.all(all_prod_ppaulis == expected_ppaulis))

        a_ppaulis = ppa.PhasedPauliArray.from_labels(["ZYYY", "XYZY", "YYZI"])
        b_ppaulis = ppa.PhasedPauliArray.from_labels(["XXXX", "YYYY", "ZZZZ"])

        expected_ppaulis = ppa.PhasedPauliArray.from_labels(
            [
                ["-YZZZ", "-jXIII", "-jIXXX"],
                ["-jIZYZ", "ZIXI", "jYXIX"],
                ["-jZZYX", "-jIIXY", "-XXIZ"],
            ]
        )

        all_prod_paulis = a_ppaulis[:, None].compose_phased_pauli_array(b_ppaulis[None, :])

        self.assertTrue(np.all(all_prod_paulis == expected_ppaulis))

    def test_str(self):
        ppaulis = ppa.PhasedPauliArray.from_labels(
            [["IIIX", "+IIIY", "-IIIZ"], ["iIIIX", "+iIIIY", "-iIIIZ"], ["jIIIX", "+jIIIY", "-jIIIZ"]]
        )

        self.assertEqual(str(ppaulis), "PhasedPauliArray: num_qubits = 4, shape = (3, 3), ...")

    def test_extract(self):
        labels1 = ["iIIIZ", "IIXX", "IYYY", "-IIIZ"]
        wpaulis = ppa.PhasedPauliArray.from_labels(labels1)

        labels2 = ["iIIIZ", "-IIIZ"]
        expected_wpaulis = ppa.PhasedPauliArray.from_labels(labels2)

        extracted_wpaulis = wpaulis.extract([True, False, False, True])

        self.assertTrue(np.all(extracted_wpaulis == expected_wpaulis))

        self.assertRaises(ValueError, lambda: wpaulis.extract([True, False, False, True, True]))

    def test_to_npz_from_npz(self):

        ppaulis = ppa.PhasedPauliArray.from_labels(
            [["IIIX", "+IIIY", "-IIIZ"], ["iIIIX", "+iIIIY", "-iIIIZ"], ["jIIIX", "+jIIIY", "-jIIIZ"]]
        )

        ppaulis.to_npz("wpaulis.npz")

        ref_ppaulis = ppa.PhasedPauliArray.from_npz("wpaulis.npz")

        self.assertTrue(np.all(ppaulis == ref_ppaulis))

        os.remove("wpaulis.npz")

    def test_to_label(self):
        ppaulis = ppa.PhasedPauliArray.from_labels(
            [["IIIX", "+IIIY", "-IIIZ"], ["iIIIX", "+iIIIY", "-iIIIZ"], ["jIIIX", "+jIIIY", "-jIIIZ"]]
        )

        ref_labels = np.array(
            [["  IIIX", "  IIIY", " -IIIZ"], [" iIIIX", " iIIIY", "-iIIIZ"], [" iIIIX", " iIIIY", "-iIIIZ"]]
        )

        assert np.all(ppaulis.to_labels() == ref_labels)


class TestPhasedPauliArrayFunc(unittest.TestCase):
    def test_concatenate(self):

        ppaulis_1 = ppa.PhasedPauliArray.from_labels(
            [["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"], ["IIIX", "IIIY", "IIIZ"]]
        )
        ppaulis_2 = ppa.PhasedPauliArray.from_labels(
            [["IIIX", "IIIX", "IIIX"], ["IIIY", "IIIY", "IIIY"], ["IIIY", "IIIY", "IIIY"]]
        )

        wpaulis_3 = ppa.concatenate((ppaulis_1, ppaulis_2), 1)

    def test_commutator(self):
        weights = np.array([1, 2, 3])

        ppaulis_1 = ppa.PhasedPauliArray.from_labels(["IIIX", "IIIY", "IIIZ"])
        ppaulis_2 = ppa.PhasedPauliArray.from_labels(["IIIY", "IIIZ", "IIIZ"])

        commutators = ppa.commutator(ppaulis_1, ppaulis_2)

    def test_anticommutator(self):
        weights = np.array([1, 2, 3])

        ppaulis_1 = ppa.PhasedPauliArray.from_labels(["IIIX", "IIIY", "IIIZ"])
        ppaulis_2 = ppa.PhasedPauliArray.from_labels(["IIIY", "IIIZ", "IIIZ"])

        anticommutators = ppa.anticommutator(ppaulis_1, ppaulis_2)


if __name__ == "__main__":
    unittest.main()
