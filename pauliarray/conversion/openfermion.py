import numpy as np
from openfermion import QubitOperator

from pauliarray import Operator, WeightedPauliArray
from pauliarray.pauli.operator_array_type_1 import OperatorArrayType1


def labels_and_weights_from_qubit_operator(qubit_op: QubitOperator, nb_qubits: int) -> tuple[list[str], list[complex]]:
    """
    Extracts labels and weights lists from a OpenFermion QubitOperator.

    Args:
        qubit_op (QubitOperator): OpenFermion QubitOperator.
        nb_qubits (int): Number of qubits on which the operator acts.

    Returns:
        tuple[list[str], list[complex]]: Lists of labels and weights (coefficients) associated with input QubitOperator.
    """
    weights = []
    labels = []
    if qubit_op.terms:
        op, weights = zip(*list(qubit_op.terms.items()))
        weights = list(weights)
        labels = []
        for label_info in op:
            label = np.array(["I"] * nb_qubits)
            if label_info:
                indices, pauli_mat = zip(*label_info)
                label[list(indices)] = list(pauli_mat)
            labels.append("".join(label)[::-1])

    return labels, weights


def qubit_operators_to_operator_array(qubit_ops: list[QubitOperator], nb_qubits: int) -> OperatorArrayType1:
    """
    Creates a PauliArray OperatorArrayType1 from a list of OpenFermion QubitOperators.

    Args:
        qubit_ops (list[QubitOperator]): List of OpenFermion QubitOperators to transform.
        nb_qubits (int): Number of qubits on which the operators act.

    Returns:
        OperatorArrayType1: Operator array containing all operators in input list.
    """
    max_length = max([len(qubit_op.terms) for qubit_op in qubit_ops])
    labels = []
    weights = []
    for qubit_op in qubit_ops:
        if qubit_op.terms:
            current_op_labels, current_op_weights = labels_and_weights_from_qubit_operator(
                qubit_op, nb_qubits, max_length
            )
            for _ in range(max_length - len(current_op_labels)):
                current_op_labels.append("".join(np.array(["I"] * nb_qubits))[::-1])
                current_op_weights.append(0.0)

            weights.append(current_op_weights)
            labels.append(current_op_labels)
    return OperatorArrayType1(WeightedPauliArray.from_labels_and_weights(labels=labels, weights=weights))


def qubit_operator_to_operator(qubit_op: QubitOperator, nb_qubits: int) -> Operator:
    """
    Creates a PauliArray Operator from a OpenFermion QubitOperator.

    Args:
        qubit_op (QubitOperator): OpenFermion QubitOperator to convert.
        nb_qubits (int): Number of qubits on which the operator acts.

    Returns:
        Operator: Operator with the same form as input QubitOperator.
    """
    labels, weights = labels_and_weights_from_qubit_operator(qubit_op=qubit_op, nb_qubits=nb_qubits)
    return Operator.from_labels_and_weights(labels=labels, weights=weights)
