# %%

import numpy as np
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import SquareLattice

from pauliarray.conversion.qiskit import extract_fermionic_op
from pauliarray.mapping.fermion import FermionMapping, JordanWigner, Parity

# %%

lattice_dim = (2, 2)
num_sites = np.prod(lattice_dim)
num_qubits = 2 * num_sites

map_matrix = np.tril(np.random.randint(0, 2, size=(num_qubits, num_qubits)), k=-1) + np.eye(num_qubits, dtype=int)
print(map_matrix)

mapper = FermionMapping(map_matrix, "random")

fermi_hubbard_model = FermiHubbardModel(SquareLattice(*lattice_dim), 1)


qiskit_fermionic_op = fermi_hubbard_model.second_q_op()

hamiltonian = mapper.assemble_qubit_hamiltonian_from_sparses(*extract_fermionic_op(qiskit_fermionic_op))

print(hamiltonian.inspect())


# %%


from qiskit_nature.second_q.mappers import JordanWignerMapper

qiskit_mapper = JordanWignerMapper()
qiskit_hamiltonian = qiskit_mapper.map(qiskit_fermionic_op)

print(qiskit_hamiltonian)

# %%
