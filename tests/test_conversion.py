import unittest

import numpy as np
from openfermion import QubitOperator
from qiskit.quantum_info import PauliList, SparsePauliOp

from pauliarray import Operator, PauliArray, WeightedPauliArray
from pauliarray.conversion.openfermion import qubit_operators_to_operator_array
from pauliarray.conversion.pennylane import operator_array_to_pauli_sentence_list
from pauliarray.conversion.qiskit import (
    operator_from_sparse_pauli,
    operator_to_sparse_pauli,
    pauli_array_from_pauli_list,
    pauli_array_to_pauli_list,
    weighted_pauli_array_from_pauli_list,
)


def get_qubit_operators_list():
    op_1 = [
        QubitOperator(term=((0, "Y"), (1, "Z"), (2, "X")), coefficient=0.5j),
        QubitOperator(term=((0, "X"), (1, "Z"), (2, "Y")), coefficient=-0.5j),
        QubitOperator(term=((1, "Y"), (2, "Z"), (3, "X")), coefficient=0.5j),
        QubitOperator(term=((1, "X"), (2, "Z"), (3, "Y")), coefficient=-0.5j),
    ]
    op_2 = [
        QubitOperator(term=((0, "Y"), (1, "Z"), (2, "Z"), (3, "Z"), (4, "X")), coefficient=0.5j),
        QubitOperator(term=((0, "X"), (1, "Z"), (2, "Z"), (3, "Z"), (4, "Y")), coefficient=-0.5j),
        QubitOperator(term=((1, "Y"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "X")), coefficient=0.5j),
        QubitOperator(term=((1, "X"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Y")), coefficient=-0.5j),
    ]
    a = op_1[0]
    b = op_2[0]
    for i in range(1, len(op_1)):
        a = QubitOperator.accumulate(a, op_1[i])
        b = QubitOperator.accumulate(b, op_2[i])
    return [a, b]


class TestPauliFromQiskit(unittest.TestCase):
    def test_pauli_array_to_pauli_list(self):
        test_pauli_list = PauliList(["IZXX", "XXII", "XXZZ"])

        paulis = PauliArray.from_labels(["IZXX", "XXII", "XXZZ"])
        pauli_list = pauli_array_to_pauli_list(paulis)

        self.assertTrue(np.all(test_pauli_list == pauli_list))

    def test_pauli_array_from_pauli_list(self):
        test_paulis = PauliArray.from_labels(["IZXX", "XXII", "XXZZ"])

        pauli_table = PauliList(["IZXX", "XXII", "XXZZ"])
        paulis = pauli_array_from_pauli_list(pauli_table)

        self.assertTrue(np.all(test_paulis == paulis))

    def test_weighted_pauli_array_from_pauli_list(self):
        test_paulis = WeightedPauliArray.from_labels_and_weights(["IZXX", "XXII", "XXZZ"], [1.0, 1j, -1j])

        pauli_table = PauliList(["IZXX", "XXII", "XXZZ"])
        pauli_table.phase = [0, 3, 5]
        paulis = weighted_pauli_array_from_pauli_list(pauli_table)

        self.assertTrue(np.all(test_paulis == paulis))

    def test_operator_from_sparse_pauli(self):
        test_operator = Operator.from_labels_and_weights(["II", "XZ"], np.array([0.5, 0.5]))

        sparse_pauli = SparsePauliOp(["II", "XZ"], np.array([0.5, 0.5]))
        operator = operator_from_sparse_pauli(sparse_pauli)

        self.assertTrue(np.all(test_operator == operator))

    def test_operator_to_sparse_pauli(self):
        test_sparse_pauli = SparsePauliOp(["II", "XZ"], np.array([0.5, 0.5]))

        operator = Operator.from_labels_and_weights(["II", "XZ"], np.array([0.5, 0.5]))
        sparse_pauli = operator_to_sparse_pauli(operator)

        self.assertTrue(np.all(test_sparse_pauli == sparse_pauli))

    # def test_qubit_operators_to_operator_array(self):
    #     qubit_ops = get_qubit_operators_list()

    #     operator_array = qubit_operators_to_operator_array(qubit_ops=qubit_ops, nb_qubits=6)

    # def test_operator_array_to_pauli_sentence_list(self):
    #     qubit_ops = get_qubit_operators_list()
    #     operator_array = qubit_operators_to_operator_array(qubit_ops=qubit_ops, nb_qubits=6)
    #     pauli_sentence_list = operator_array_to_pauli_sentence_list(operator_array)


if __name__ == "__main__":
    unittest.main()
