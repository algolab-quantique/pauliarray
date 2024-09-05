import numpy as np

import pauliarray.binary.void_operations as vops
import pauliarray.pauli.pauli_array as pa


def gen_complete_pauli_array_basis(num_qubits: int) -> pa.PauliArray:
    """
    Generates a PauliArray containining all the Pauli strings for n qubits.

    Args:
        num_qubits (int): The number of qubits

    Returns:
        PauliArray: A complete basis
    """
    all_ints = np.arange(2**num_qubits, dtype=np.uint8)

    z_powers_of_two = vops.int_strings_to_voids(
        np.broadcast_to(all_ints[None, :, None], (2**num_qubits, 2**num_qubits, 1))
    )
    x_powers_of_two = vops.int_strings_to_voids(
        np.broadcast_to(all_ints[:, None, None], (2**num_qubits, 2**num_qubits, 1))
    )

    return pa.PauliArray(z_powers_of_two, x_powers_of_two, num_qubits).flatten()


def gen_random_pauli_array(shape, number_of_qubits: int) -> pa.PauliArray:
    """
    Generates random Pauli strings

    Args:
        shape (_type_): The shape of the PauliArray
        number_of_qubits (int): The number of qubits

    Returns:
        pa.PauliArray: _description_
    """
    z_strings = np.random.choice(a=[False, True], size=shape + (number_of_qubits,))
    x_strings = np.random.choice(a=[False, True], size=shape + (number_of_qubits,))

    return pa.PauliArray.from_z_strings_and_x_strings(z_strings, x_strings)
