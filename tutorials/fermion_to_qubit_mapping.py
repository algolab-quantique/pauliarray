# %%
import time

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from pauliarray.conversion.qiskit import extract_fermionic_op, operator_to_sparse_pauli
from pauliarray.mapping.fermion import FermionMapping, JordanWigner

# %% Fermionic Hamiltonian using qiskit

mol_info = {
    "atom": "N 0 0 -0.545;N 0 0 0.545;",
    "basis": "sto3g",
    "charge": 0,
    "spin": 0,
}

driver = PySCFDriver(**mol_info)
problem = driver.run()
hamiltonian = problem.hamiltonian
second_q_hamiltonian = problem.hamiltonian.second_q_op()


# %% Fermion to qubit mapping with qiskit

t0 = time.time()

mapper = JordanWignerMapper()
qk_jw_qubit_hamiltonian = mapper.map(second_q_hamiltonian)

print(f"Qiskit : {time.time() - t0:.3f} sec")
print(f"Number of qubits : {qk_jw_qubit_hamiltonian.num_qubits}")
print(f"Number of Pauli strings : {len(qk_jw_qubit_hamiltonian)}")

# %% Fermion to qubit mapping with PauliArray

num_spin_orbitals = second_q_hamiltonian.num_spin_orbitals

t0 = time.time()

one_body_tuple, two_body_tuple = extract_fermionic_op(second_q_hamiltonian)

mapping = JordanWigner(num_spin_orbitals)
pa_jw_qubit_hamiltonien = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

print(f"PauliArray : {time.time() - t0:.3f} sec")
print(f"Number of qubits : {pa_jw_qubit_hamiltonien.num_qubits}")
print(f"Number of Pauli strings : {pa_jw_qubit_hamiltonien.num_terms}")


# %% Custom Fermion to qubit mapping with PauliArray

print(operator_to_sparse_pauli(pa_jw_qubit_hamiltonien).sort() == qk_jw_qubit_hamiltonian.sort())

mol_info = {
    "atom": "Li 0 0 0;H 0 0 1.6;",
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

print(f"{num_spin_orbitals=}")

mapping_matrix = np.eye(num_spin_orbitals, dtype=int) + np.tril(
    np.random.randint(0, 2, (num_spin_orbitals, num_spin_orbitals)), k=-1
)

print(mapping_matrix)

# %%

rd_mapping = FermionMapping(mapping_matrix)
pa_rd_qubit_hamiltonien = rd_mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)


jw_mapping = JordanWigner(num_spin_orbitals)
pa_jw_qubit_hamiltonien = jw_mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

# %%

eigvals_jw = np.linalg.eigvals(pa_jw_qubit_hamiltonien.to_matrix())
eigvals_rd = np.linalg.eigvals(pa_rd_qubit_hamiltonien.to_matrix())

print(np.all(np.sort(eigvals_jw) == np.sort(eigvals_rd)))

# %%
