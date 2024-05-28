from pennylane.pauli import PauliSentence, PauliWord

from pauliarray import OperatorArrayType2


def operator_array_to_pauli_sentence_list(operator_array: OperatorArrayType2) -> list[PauliSentence]:
    """
    Creates a list of Pennylane PauliSentences from an OperatorArrayType2.

    Args:
        operator_array (OperatorArrayType2): Input OperatorArray to convert into list of PauliSentences.

    Returns:
        list[PauliSentence]: PauliSentences obtained from input OperatorArray.
    """
    pauli_sentence_list = []
    for operator in operator_array:
        wpa = operator.wpaulis
        pauli_s = {}
        for l, w in list(zip(wpa.paulis.to_labels(), wpa.weights)):
            label_dict = {}
            for qubit, pauli_mat in enumerate(l):
                label_dict[wpa.size - 1 - qubit] = pauli_mat
            pauli_s[PauliWord(label_dict)] = w
        pauli_sentence_list.append(PauliSentence(pauli_s))
    return pauli_sentence_list
