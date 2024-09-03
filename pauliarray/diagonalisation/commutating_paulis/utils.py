import numpy as np

import pauliarray.pauli.pauli_array as pa


def trivial_cummutating_generators(paulis: pa.PauliArray):

    num_qubits = paulis.num_qubits

    gen_z_strings = np.zeros((num_qubits, num_qubits), dtype=bool)
    gen_x_strings = np.zeros((num_qubits, num_qubits), dtype=bool)

    for q in range(paulis.num_qubits):
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
