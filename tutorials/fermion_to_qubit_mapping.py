# %%
import time

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from pauliarray.conversion.qiskit import extract_fermionic_op, operator_to_sparse_pauli
from pauliarray.mapping.fermion import FermionMapping, JordanWigner

# In this tutorial we will see how to map a fermionic Hamiltonian to a qubit Hamiltonian using PauliArray. We will also make a comparison with similar tools provided in Qiskit and show that PauliArray accomplishes the same task much faster.

# %% Fermionic Hamiltonian using qiskit

# As a starting point, we will use Qiskit to generate the fermionic Hamiltonian (:code:`FermioncOp`) for the :math:`\text{N}_2` molecule.

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


# %% Jordan Wigner Mapping using Qiskit

# Carrying out the mapping with Qiskit is pretty straight forward. The result is a :code:`SparsePauliOp`. We can measure the time it takes to complete the process.

t0 = time.time()

mapper = JordanWignerMapper()
qk_jw_qubit_hamiltonian = mapper.map(second_q_hamiltonian)

print(f"Qiskit : {time.time() - t0:.3f} sec")
print(f"Number of qubits : {qk_jw_qubit_hamiltonian.num_qubits}")
print(f"Number of Pauli strings : {len(qk_jw_qubit_hamiltonian)}")

# %% Jordan Wigner Mapping using PauliArray

# The process is pretty similar using PauliArray except we need to convert the :code:`FermioncOp` into arguments compatible with the :code:`FermionMapping`. We also need to specify the number of qubits to initialize the :code:`JordanWigner` mapping. The result is a :code:`Operator`.

# We can check that both result are the same by converting the :code:`Operator` into a :code:`SparsePauliOp`.

num_spin_orbitals = second_q_hamiltonian.num_spin_orbitals

t0 = time.time()

one_body_tuple, two_body_tuple = extract_fermionic_op(second_q_hamiltonian)

mapping = JordanWigner(num_spin_orbitals)
pa_jw_qubit_hamiltonien = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

print(f"PauliArray : {time.time() - t0:.3f} sec")
print(f"Number of qubits : {pa_jw_qubit_hamiltonien.num_qubits}")
print(f"Number of Pauli strings : {pa_jw_qubit_hamiltonien.num_terms}")


print(operator_to_sparse_pauli(pa_jw_qubit_hamiltonien).sort() == qk_jw_qubit_hamiltonian.sort())


# %% Custom Fermion to qubit mapping with PauliArray

# PauliArray allows for constructing mapping for :math:`n` states by providing an invertible binary component :math:`n\times n` matrix. To show this, we will consider a smaller molecule :math:`\text{LiH}`.

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

# %%

# Let's construct such a matrix randomly. Noteworthy, such a matrix with 1 on the diagonal, an upper triangle filled with 0, and a random lower triangle is guaranteed to be invertible.

# To initialize the mapping, we only need to provide this matrix to :code:`FermionMapping`. The mapping is then used in the same way as before to construct a qubit Hamiltonian.

# Finally, to confirm that such a mapping is valid we can compare the qubit Hamiltonian it produces with the one we get from Jordan-Wigner mapping. These two Hamiltonians are expressing the same operator but in different basis. Therefore, their eigenvalues should be equals.

mapping_matrix = np.eye(num_spin_orbitals, dtype=int) + np.tril(
    np.random.randint(0, 2, (num_spin_orbitals, num_spin_orbitals)), k=-1
)

print(mapping_matrix)

rd_mapping = FermionMapping(mapping_matrix)
pa_rd_qubit_hamiltonien = rd_mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)


jw_mapping = JordanWigner(num_spin_orbitals)
pa_jw_qubit_hamiltonien = jw_mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

eigvals_jw = np.linalg.eigvals(pa_jw_qubit_hamiltonien.to_matrix())
eigvals_rd = np.linalg.eigvals(pa_rd_qubit_hamiltonien.to_matrix())

print(np.all(np.sort(eigvals_jw) == np.sort(eigvals_rd)))

# %%
