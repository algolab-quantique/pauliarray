# %%

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver

from pauliarray.conversion.qiskit import extract_fermionic_op
from pauliarray.diagonalisation.commutating_paulis.with_circuits import (
    general_to_diagonal as general_to_diagonal_with_circuit,
)
from pauliarray.diagonalisation.commutating_paulis.with_operators import (
    general_to_diagonal as general_to_diagonal_with_operator,
)
from pauliarray.mapping.fermion import BravyiKitaev, JordanWigner, Parity
from pauliarray.partition.commutating_paulis.exclusive_fct import (  # partition_same_z,
    partition_general_commutating,
    partition_same_x,
    partition_same_x_plus_special,
)

# %%

mol_info = {
    "atom": "N 0 0 -0.545;N 0 0 0.545;",
    # "atom": "Li 0 0 0;H 0 0 1.6;",
    "basis": "sto3g",
    "charge": 0,
    "spin": 0,
}

driver = PySCFDriver(**mol_info)
problem = driver.run()
hamiltonian = problem.hamiltonian
second_q_hamiltonian = problem.hamiltonian.second_q_op()

num_spin_orbitals = second_q_hamiltonian.num_spin_orbitals

one_body_tuple, two_body_tuple = extract_fermionic_op(second_q_hamiltonian)

mapping = JordanWigner(num_spin_orbitals)
# mapping = Parity(num_spin_orbitals)
# mapping = BravyiKitaev(num_spin_orbitals)
qubit_hamiltonien = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

# %%

hamiltonian_parts = qubit_hamiltonien.partition_with_fct(partition_same_x_plus_special)
num_parts = len(hamiltonian_parts)

print(f"{num_parts=}")

# %%

diag_parts = []
factors_parts = []
transformations_parts = []
transformation_num_active_qubits_parts = []

for i, part in enumerate(hamiltonian_parts):

    diag_part, factors_part, transformations_part_operators = general_to_diagonal_with_operator(
        part.paulis, force_trivial_generators=True
    )

    diag_part, factors_part, transformations_part_circuits = general_to_diagonal_with_circuit(part.paulis)

    # transformation_num_active_qubits = np.max(
    #     np.sum(
    #         np.logical_or(
    #             transformations_part_operators.paulis.z_strings, transformations_part_operators.paulis.x_strings
    #         ),
    #         axis=-1,
    #     ),
    #     axis=-1,
    # )


#     if np.any(transformation_num_active_qubits == 7):
#         print(part.inspect())
#         print(diag_part.inspect())

#     transformation_num_active_qubits_parts.append(transformation_num_active_qubits)

#     diag_parts.append(diag_parts)
#     factors_parts.append(factors_parts)
#     transformations_parts.append(transformations_parts)

# transformation_num_active_qubits = np.concatenate(transformation_num_active_qubits_parts)

# print(np.bincount(transformation_num_active_qubits))

# %%
