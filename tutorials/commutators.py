# %%


import numpy as np

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa

# PauliArray is a fast and convenient tool to compute commutators between operators. This tutorial looks at this in the context of the adiabatic evolution applied to an Ising problem.

# %% Ising Hamiltonian

# The system we will be considering is an ensemble of spins on a graph defined by the following edges.

edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
    (0, 2),
]
number_of_nodes = 5
number_of_edges = len(edges)

# We define the Hamiltonian with the Ising interaction (ZZ) on that graph


z_strings = np.zeros((number_of_edges, number_of_nodes), dtype=bool)
x_strings = np.zeros((number_of_edges, number_of_nodes), dtype=bool)

for i_edge, edge in enumerate(edges):
    z_strings[i_edge, edge] = True

ising_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(z_strings, x_strings), 0.5)

print(ising_hamiltonian.inspect())

# %% Drive Hamiltonian

# When trying to find the ground state of a Hamiltonian such as :math:`\hat{H}_\text{Ising}`, the quantum annealing method consists of starting in the ground state of an easily solvable Hamiltonian, also known as a drive Hamiltonian in certain contexts. We select this Hamiltonian as a transverse X field on all qubits.

z_strings = np.zeros((number_of_nodes, number_of_nodes), dtype=bool)
x_strings = np.eye(number_of_nodes, dtype=bool)

drive_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(z_strings, x_strings), -1)

print(drive_hamiltonian.inspect())

# %% Commutator

# Each data structure comes with its own function to compute commutators. Here we use one to compute the hermitian commutator (with a factor 1j) between :math:`\hat{H}_\text{Ising}` and :math:`\hat{H}_\text{Drive}`.

commutator = 1j * op.commutator(ising_hamiltonian, drive_hamiltonian)

print(commutator.inspect())

# %% Individual Commutators

# PauliArray makes it easy to access all the individual commutators between the Pauli strings of :math:`\hat{H}_\text{Ising}` and :math:`\hat{H}_\text{Drive}`

commutators, factor = pa.commutator(ising_hamiltonian.paulis[:, None], drive_hamiltonian.paulis[None, :])

print(commutators.inspect())
