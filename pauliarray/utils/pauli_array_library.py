import numpy as np

import pauliarray.pauli.pauli_array as pa


def gen_complete_pauli_array_basis(number_of_qubits: int) -> pa.PauliArray:
    """
    Generates a PauliArray containining all the Pauli strings for n qubits.

    Args:
        number_of_qubits (int): The number of qubits

    Returns:
        PauliArray: A complete basis
    """
    bin_power = 2 ** np.arange(number_of_qubits, dtype=np.uintc)
    bits = ((np.arange(2 ** (number_of_qubits), dtype=np.uintc)[:, None] & bin_power[None, :]) > 0).reshape(
        (2**number_of_qubits, number_of_qubits)
    )
    z_bits = np.broadcast_to(bits[None, :], (2**number_of_qubits, 2**number_of_qubits, number_of_qubits))
    x_bits = np.broadcast_to(bits[:, None], (2**number_of_qubits, 2**number_of_qubits, number_of_qubits))
    return pa.PauliArray(z_bits, x_bits).flatten()


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

    return pa.PauliArray(z_strings, x_strings)
