# %%

import numpy as np
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import SquareLattice

from pauliarray.conversion.qiskit import extract_fermionic_op
from pauliarray.mapping.fermion import BravyiKitaev, FermionMapping, JordanWigner, Parity

# %%

lattice_dim = (2, 2)
num_sites = np.prod(lattice_dim)
num_qubits = 2 * num_sites

map_matrix = np.tril(np.random.randint(0, 2, size=(num_qubits, num_qubits)), k=-1) + np.eye(num_qubits, dtype=int)
# print(map_matrix)

mapper = FermionMapping(map_matrix, "random")
# mapper = Parity(num_qubits)
# mapper = JordanWigner(num_qubits)
# mapper = BravyiKitaev(num_qubits)

fermi_hubbard_model = FermiHubbardModel(SquareLattice(*lattice_dim), 1)

qiskit_fermionic_op = fermi_hubbard_model.second_q_op()

hamiltonian = mapper.assemble_qubit_hamiltonian_from_sparses(*extract_fermionic_op(qiskit_fermionic_op))
# print(hamiltonian.inspect())

pauli_weights, counts = np.unique(hamiltonian.paulis.num_non_ids, return_counts=True)
order = np.argsort(pauli_weights)
pauli_weights, counts = pauli_weights[order], counts[order]
for pauli_weight, count in zip(pauli_weights, counts):
    print(pauli_weight, count)


# %%


from qiskit_nature.second_q.mappers import JordanWignerMapper

qiskit_mapper = JordanWignerMapper()
qiskit_hamiltonian = qiskit_mapper.map(qiskit_fermionic_op)

print(qiskit_hamiltonian)

# %%
