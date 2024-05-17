========================
Fermion to Qubit Mapping
========================

In this tutorial we will see how to map a fermionic Hamiltonian to a qubit Hamiltonian using PauliArray. We will also make a comparaison with similar tools provided in qiskit and show that PauliArray accomplish the same task much faster.

---------------------------------
Fermionic Hamiltonian with qiskit
---------------------------------

As a starting point we will use qiskit to generate the fermionic Hamiltonian (:code:`FermioncOp`) for the :math:`\text{N}_2` molecule.

.. code:: python

    from qiskit_nature.second_q.drivers import PySCFDriver

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

---------------------------------
Jordan Wigner Mapping with qiskit
---------------------------------

Carrying out the mapping with qiskit is pretty straight forward. The result is a :code:`SparsePauliOp`. We can measure the time it takes to complete the process.

.. code:: python

    import time

    from qiskit_nature.second_q.mappers import JordanWignerMapper

    t0 = time.time()

    mapper = JordanWignerMapper()
    qk_qubit_hamiltonian = mapper.map(second_q_hamiltonian)

    print(f"Qiskit : {time.time() - t0:.3f} sec")
    print(f"Number of qubits : {qk_qubit_hamiltonian.num_qubits}")
    print(f"Number of Pauli strings : {len(qk_qubit_hamiltonian)}")

.. code::

    Qiskit : 2.049 sec
    Number of qubits : 20
    Number of Pauli strings : 2951

As a not so rigourous benchmark, it takes about 2.1 sec to an Apple M2 to complete the mapping for the :math:`\text{N}_2` molecule involving 20 qubits and 2951 Pauli strings.

-------------------------------------
Jordan Wigner Mapping with PauliArray
-------------------------------------

The process is pretty similar using PauliArray, except we need to convert the :code:`FermioncOp` into arguments compatible with the :code:`FermionMapping`. We also need to specify the number of qubits to initialise the :code:`JordanWigner` mapping. The result is a :code:`Operator`. 

.. code:: python

    from pauliarray.interface.qiskit import extract_fermionic_op
    from pauliarray.mapping.fermion import JordanWigner

    num_spin_orbitals = second_q_hamiltonian.num_spin_orbitals

    t0 = time.time()

    one_body_tuple, two_body_tuple = extract_fermionic_op(second_q_hamiltonian)

    mapping = JordanWigner(num_spin_orbitals)
    pa_qubit_hamiltonien = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

    print(f"PauliArray : {time.time() - t0:.3f} sec")
    print(f"Number of qubits : {pa_qubit_hamiltonien.num_qubits}")
    print(f"Number of Pauli strings : {pa_qubit_hamiltonien.num_terms}")

.. code::

    PauliArray : 0.124 sec
    Number of qubits : 20
    Number of Pauli strings : 2951

Again, the not so rigourous benchmarking shows that PauliArray completes the mapping in less than 0.15 sec, which is more than 10 times faster than qiskit.

We can check that both result are the same by converting the :code:`Operator` into a :code:`SparsePauliOp`. 

.. code:: python

    from pauliarray.interface.qiskit import operator_to_sparse_pauli

    print(operator_to_sparse_pauli(pa_qubit_hamiltonien).sort() == qk_qubit_hamiltonian.sort())

.. code::

    True

----------------------------------------
General (Random) Mapping with PauliArray
----------------------------------------

PauliArray allow to construct mapping for :math:`n` states by providing an invertible binary component :math:`n\times n` matrix. To show this, we will consider a smaller molecule :math:`LiH`.

.. code::

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

.. code::

    num_spin_orbitals=12

Let's construct such a matrix randomly.

.. note::

    Such a matrix with 1 on the diagonal, an upper triangle filled with 0, and a random lower triangle is garanteed to be invertible.

.. code:: python

    mapping_matrix = np.eye(num_spin_orbitals, dtype=int) + np.tril(
        np.random.randint(0, 2, (num_spin_orbitals, num_spin_orbitals)), k=-1
    )

    print(mapping_matrix)

.. code::

    [[1 0 0 0 0 0 0 0 0 0 0 0]
     [1 1 0 0 0 0 0 0 0 0 0 0]
     [1 1 1 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 0 0 0]
     [0 0 1 1 1 0 0 0 0 0 0 0]
     [0 1 0 0 0 1 0 0 0 0 0 0]
     [0 1 1 0 1 1 1 0 0 0 0 0]
     [1 1 1 0 1 0 0 1 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 0 0 0]
     [0 0 1 1 1 1 0 1 0 1 0 0]
     [1 0 0 1 1 0 1 0 0 0 1 0]
     [0 0 1 0 1 0 1 0 1 1 1 1]]

To initialise the mapping, we only need to provide this matrix to :code:`FermionMapping`.

.. code:: python

    from pauliarray.mapping.fermion import FermionMapping    

    mapping = FermionMapping(mapping_matrix)
    pa_qubit_hamiltonien = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

Finally, to validate that such mapping are valid, we can contruct two random mappings. Their respective mapping matrices should be different. Then we can use them to convert the fermionic Hamiltonian to two different qubit Hamiltonians. These two Hamiltonian are expressing the same operator but in different basis. Therefore, their eigenvalues should be equals. 

Let's check that this is true. This may take a while.

.. code:: python

    mapping_matrix_1 = np.eye(num_spin_orbitals, dtype=int) + np.tril(
    np.random.randint(0, 2, (num_spin_orbitals, num_spin_orbitals)), k=-1
    )
    mapping_matrix_2 = np.eye(num_spin_orbitals, dtype=int) + np.tril(
        np.random.randint(0, 2, (num_spin_orbitals, num_spin_orbitals)), k=-1
    )

    print(~np.all(mapping_matrix_1 == mapping_matrix_2))

    mapping_1 = FermionMapping(mapping_matrix_1)
    qubit_hamiltonien_1 = mapping_1.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)
    mapping_2 = FermionMapping(mapping_matrix_2)
    qubit_hamiltonien_2 = mapping_2.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

    eigvals_1 = np.linalg.eigvals(pa_qubit_hamiltonien.to_matrix())
    eigvals_2 = np.linalg.eigvals(pa_qubit_hamiltonien.to_matrix())

    print(np.all(np.sort(eigvals_1) == np.sort(eigvals_2)))

.. code::

    True
    True

.. Add description for BCS Hamiltonian.