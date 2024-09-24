import numpy as np

import pauliarray.pauli.pauli_array as pa


def single_qubit_cummutating_generators(paulis: pa.PauliArray):
    """
    Identify single qubit commutation generators in a list of Pauli strings. Such a generator exist if a qubit is acted upon at most by the identity and a single Pauli for every Pauli string.

    Args:
        paulis (pa.PauliArray): _description_

    Returns:
        _type_: _description_
    """

    assert paulis.ndim == 1

    num_qubits = paulis.num_qubits

    gen_z_strings = np.zeros((num_qubits, num_qubits), dtype=bool)
    gen_x_strings = np.zeros((num_qubits, num_qubits), dtype=bool)

    for q in range(num_qubits):
        u_qubit_zx = np.unique(paulis.zx_strings[:, [q, num_qubits + q]], axis=0)
        if u_qubit_zx.shape[0] == 2 and np.all(u_qubit_zx[0] == 0):
            gen_z_strings[q, q] = u_qubit_zx[1, 0]
            gen_x_strings[q, q] = u_qubit_zx[1, 1]
        elif u_qubit_zx.shape[0] == 1:
            if np.all(u_qubit_zx[0] == 0):
                gen_z_strings[q, q] = True
            else:
                gen_z_strings[q, q] = u_qubit_zx[0, 0]
                gen_x_strings[q, q] = u_qubit_zx[0, 1]

    return pa.PauliArray(gen_z_strings, gen_x_strings)
