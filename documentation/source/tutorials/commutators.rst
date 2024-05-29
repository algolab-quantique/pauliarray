========================
Commutators
========================

PauliArray is a fast and convenient tool to compute commutators between operators. This tutorial look at this in the context of the adiabatic evolution applied to an Ising problem.


---------------------------------
Ising Hamiltonian
---------------------------------

The system we will be considering at is an ensemble of spins on a graph defined by the following edges.

.. code:: python

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

We define an Hamiltonian such that pairs of spins (connected by an edge in :math:`E`) interacts via :math:`\tfrac{1}{2}\hat{Z}\hat{Z}`. We can write this Hamiltonian 

.. math::

    \hat{H}_\text{Ising} = \frac{1}{2} \sum_{i,j \in E} \hat{Z}_i \hat{Z}_j = \frac{1}{2} \sum_{k} \hat{P}^{(I)}_k

which can also be expressed as a linear combinaison of Pauli strings :math:`\hat{P}^{(I)}_k`.

This kind of Hamiltonian is related to the MaxCut problem, often used as an example in the context of quantum combinatorial optimization.

With PauliArray, we can construct this Hamiltonian directly from the bit strings.

.. code:: python 

    import numpy as np

    import pauliarray.pauli.operator as op
    import pauliarray.pauli.pauli_array as pa

    z_strings = np.zeros((number_of_edges, number_of_nodes), dtype=bool)
    x_strings = np.zeros((number_of_edges, number_of_nodes), dtype=bool)

    for i_edge, edge in enumerate(edges):
        z_strings[i_edge, edge] = True

    ising_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(z_strings, x_strings), 0.5)

    print(ising_hamiltonian.inspect())

.. code::

    Operator
    Sum of
    (+0.5000 +0.0000j) IIIZZ
    (+0.5000 +0.0000j) IIZZI
    (+0.5000 +0.0000j) IZZII
    (+0.5000 +0.0000j) ZZIII
    (+0.5000 +0.0000j) ZIIIZ
    (+0.5000 +0.0000j) IIZIZ


---------------------------------
Drive Hamiltonian
---------------------------------

When trying to find the ground state of a Hamiltonian such as :math:`\hat{H}_\text{Ising}`, the quantum annealing method consists of starting in the ground state of an easily solvable Hamiltonian, also known as a drive Hamiltonian in certain contexts, such as

.. math::

    \hat{H}_\text{Drive} = -\sum_{i} \hat{X}_i = - \sum_{k} \hat{P}^{(D)}_k
    .

The ground state of this Hamltonian is :math:`\ket{+}^{\otimes n}`. This Hamiltonian can be constructed in the following way.

.. code:: python

    z_strings = np.zeros((number_of_nodes, number_of_nodes), dtype=bool)
    x_strings = np.eye(number_of_nodes, dtype=bool)

    drive_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(z_strings, x_strings), -1)

    print(drive_hamiltonian.inspect())

.. code::

    Operator
    Sum of
    (-1.0000 +0.0000j) IIIIX
    (-1.0000 +0.0000j) IIIXI
    (-1.0000 +0.0000j) IIXII
    (-1.0000 +0.0000j) IXIII
    (-1.0000 +0.0000j) XIIII


---------------------------------
Commutator
---------------------------------

Each data structure comes with its own function to compute commutators. Here we use one to compute the commutator between :math:`\hat{H}_\text{Ising}` and :math:`\hat{H}_\text{Drive}`

.. math::

    [\hat{H}_\text{Ising}, \hat{H}_\text{Drive}]
    .

.. code:: python

    commutator = 1j * op.commutator(ising_hamiltonian, drive_hamiltonian)

    print(commutator.inspect())


.. code::

    Operator
    Sum of
    (+1.0000 +0.0000j) IIIZY
    (+1.0000 +0.0000j) IIIYZ
    (+1.0000 +0.0000j) IIZYI
    (+1.0000 +0.0000j) IIYZI
    (+1.0000 +0.0000j) IZYII
    (+1.0000 +0.0000j) IYZII
    (+1.0000 +0.0000j) ZYIII
    (+1.0000 +0.0000j) YZIII
    (+1.0000 +0.0000j) ZIIIY
    (+1.0000 +0.0000j) YIIIZ
    (+1.0000 +0.0000j) IIZIY
    (+1.0000 +0.0000j) IIYIZ


---------------------------------
Individual Commutators 
---------------------------------

PauliArray makes it easy to access all the individual commutators between the Pauli strings of :math:`\hat{H}_\text{Ising}` and :math:`\hat{H}_\text{Drive}`

.. math::
    
    [\hat{H}_\text{Ising}, \hat{H}_\text{Drive}] = -\frac{1}{2} \sum_{k,l} [\hat{P}^{(I)}_k, \hat{P}^{(D)}_l]
    .

.. code:: python

    commutators, factor = pa.commutator(ising_hamiltonian.paulis[:, None], drive_hamiltonian.paulis[None, :])

    print(commutators.inspect())


.. code::

    PauliArray
    IIIZY  IIIYZ  IIIII  IIIII  IIIII
    IIIII  IIZYI  IIYZI  IIIII  IIIII
    IIIII  IIIII  IZYII  IYZII  IIIII
    IIIII  IIIII  IIIII  ZYIII  YZIII
    ZIIIY  IIIII  IIIII  IIIII  YIIIZ
    IIZIY  IIIII  IIYIZ  IIIII  IIIII