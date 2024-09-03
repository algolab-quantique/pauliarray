from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from scipy.optimize import linear_sum_assignment

import pauliarray.pauli.pauli_array as pa
from pauliarray.binary.bit_operations import pack_diagonal
from pauliarray.diagonalisation.commutating_paulis.utils import trivial_cummutating_generators


def general_to_diagonal(
    paulis: pa.PauliArray, force_trivial_generators=False
) -> Tuple[pa.PauliArray, NDArray[np.complex_], List[str]]:

    circuit = QuantumCircuit(paulis.num_qubits)

    generators_paulis = paulis.generators()

    assert generators_paulis.ndim == 1

    if force_trivial_generators:
        trivial_generators_paulis = trivial_cummutating_generators(generators_paulis)
        generators_paulis = pa.concatenate((generators_paulis, trivial_generators_paulis), axis=0)

    wpaulis = paulis.wpaulis
    qubit_order = np.arange(paulis.num_qubits)

    x_table = generators_paulis.x_strings
    z_table = generators_paulis.z_strings

    # diagonalisation of the X block
    x_table, row_op, col_order, start_index = pack_diagonal(x_table, 0)
    z_table = np.mod((row_op.astype(np.uint) @ z_table.astype(np.uint)), 2).astype(bool)[:, col_order]
    qubit_order = qubit_order[col_order]

    end_block_x = start_index

    # diagonalisation of the Z block
    z_table, row_op, col_order, start_index = pack_diagonal(z_table, start_index)
    x_table = np.mod((row_op.astype(np.uint) @ x_table.astype(np.uint)), 2).astype(bool)[:, col_order]
    qubit_order = qubit_order[col_order]

    end_block_z = start_index

    # apply hadamard

    tmp = x_table[:, end_block_x:end_block_z].copy()
    x_table[:, end_block_x:end_block_z] = z_table[:, end_block_x:end_block_z].copy()
    z_table[:, end_block_x:end_block_z] = tmp

    wpaulis.h(qubit_order[end_block_x:end_block_z])
    circuit.h(qubit_order[end_block_x:end_block_z])

    # print(np.all(wpaulis[:, None].commute_with(wpaulis[None, :])))

    # clear X with CNOT
    for i in range(end_block_z):
        if np.any(x_table[i, i + 1 :]):
            _js = np.where(x_table[i, i + 1 :])[0] + i + 1
            _is = i * np.ones(_js.shape, dtype=int)
            x_table[:, _js] = np.logical_xor(x_table[:, _js], x_table[:, _is])
            z_table[:, i] = np.logical_xor(z_table[:, i], np.mod(np.sum(z_table[:, _js], axis=1), 2).astype(bool))

            wpaulis.cx(qubit_order[_is], qubit_order[_js])
            circuit.cx(qubit_order[_is], qubit_order[_js])

    # clear Z block with CZ
    for i in range(1, end_block_z):
        if np.any(z_table[i, :i]):
            _js = np.where(z_table[i, :i])[0]
            _is = i * np.ones(_js.shape, dtype=int)
            z_table[:, _js] = np.logical_xor(z_table[:, _js], x_table[:, _is])
            z_table[:, i] = np.logical_xor(z_table[:, i], np.mod(np.sum(x_table[:, _js], axis=1), 2).astype(bool))

            wpaulis.cz(qubit_order[_is], qubit_order[_js])
            circuit.cz(qubit_order[_is], qubit_order[_js])

    # clear diag Z with S
    _is = np.where(z_table.diagonal()[:end_block_z])[0]
    z_table[_is, _is] = False
    if _is:
        wpaulis.s(qubit_order[_is])
        circuit.s(qubit_order[_is])

    # clear all diag X with H
    _is = np.arange(end_block_z)
    x_table[_is, _is] = False
    z_table[_is, _is] = True
    wpaulis.h(qubit_order[np.arange(end_block_z)])
    circuit.h(qubit_order[np.arange(end_block_z)])

    # print()
    # print("x")
    # print(x_table.astype(int))
    # print("z")
    # print(z_table.astype(int))

    # print(wpaulis.inspect())

    # work_generators = pa.PauliArray(z_table, x_table)
    # assert np.all(work_generators[:, None].commute_with(work_generators[None, :]))

    diag_paulis = wpaulis.paulis
    factors = wpaulis.weights

    return diag_paulis, factors, circuit
