========================
Fermion to Qubit Mapping
========================

In this tutorial we will see how to map a fermionic Hamiltonian to a qubit Hamiltonian using PauliArray. We will also make a comparison with similar tools provided in Qiskit and show that PauliArray accomplish the same task much faster.

---------------------------------
Fermionic Hamiltonian with Qiskit
---------------------------------

As a starting point, we will use Qiskit to generate the fermionic Hamiltonian (:code:`FermioncOp`) for the :math:`\text{N}_2` molecule.

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
Jordan Wigner Mapping with Qiskit
---------------------------------

Carrying out the mapping with Qiskit is pretty straight forward. The result is a :code:`SparsePauliOp`. We can measure the time it takes to complete the process.

.. code:: python

    import time

    from qiskit_nature.second_q.mappers import JordanWignerMapper

    t0 = time.time()

    mapper = JordanWignerMapper()
    qk_jw_qubit_hamiltonian = mapper.map(second_q_hamiltonian)

    print(f"Qiskit : {time.time() - t0:.3f} sec")
    print(f"Number of qubits : {qk_jw_qubit_hamiltonian.num_qubits}")
    print(f"Number of Pauli strings : {len(qk_jw_qubit_hamiltonian)}")

.. code::

    Qiskit : 2.049 sec
    Number of qubits : 20
    Number of Pauli strings : 2951

As a not so rigorous benchmark, it takes about 2.1 sec to an Apple M2 to complete the mapping for the :math:`\text{N}_2` molecule involving 20 qubits and 2951 Pauli strings.

-------------------------------------
Jordan Wigner Mapping with PauliArray
-------------------------------------

The process is pretty similar using PauliArray except we need to convert the :code:`FermioncOp` into arguments compatible with the :code:`FermionMapping`. We also need to specify the number of qubits to initialize the :code:`JordanWigner` mapping. The result is a :code:`Operator`.

.. code:: python

    from pauliarray.conversion.Qiskit import extract_fermionic_op
    from pauliarray.mapping.fermion import JordanWigner

    num_spin_orbitals = second_q_hamiltonian.num_spin_orbitals

    t0 = time.time()

    one_body_tuple, two_body_tuple = extract_fermionic_op(second_q_hamiltonian)

    mapping = JordanWigner(num_spin_orbitals)
    pa_jw_qubit_hamiltonien = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

    print(f"PauliArray : {time.time() - t0:.3f} sec")
    print(f"Number of qubits : {pa_jw_qubit_hamiltonien.num_qubits}")
    print(f"Number of Pauli strings : {pa_jw_qubit_hamiltonien.num_terms}")

.. code::

    PauliArray : 0.124 sec
    Number of qubits : 20
    Number of Pauli strings : 2951

Again, the not so rigorous benchmarking shows that PauliArray completes the mapping in less than 0.15 sec, which is more than 10 times faster than Qiskit.

We can check that both result are the same by converting the :code:`Operator` into a :code:`SparsePauliOp`.

.. code:: python

    from pauliarray.conversion.Qiskit import operator_to_sparse_pauli

    print(operator_to_sparse_pauli(pa_jw_qubit_hamiltonien).sort() == qk_jw_qubit_hamiltonian.sort())

.. code::

    True

----------------------------------------
General (Random) Mapping with PauliArray
----------------------------------------

PauliArray allow for constructing mapping for :math:`n` states by providing an invertible binary component :math:`n\times n` matrix. To show this, we will consider a smaller molecule :math:`\text{LiH}`.

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

    Such a matrix with 1 on the diagonal, an upper triangle filled with 0, and a random lower triangle is guaranteed to be invertible.

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

To initialize the mapping, we only need to provide this matrix to :code:`FermionMapping`. The mapping is then used in the same way as before to construct a qubit Hamiltonian.

.. code:: python

    from pauliarray.mapping.fermion import FermionMapping

    mapping = FermionMapping(mapping_matrix)
    pa_rd_qubit_hamiltonien = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

Finally, to confirm that such a mapping is valid we can compare the qubit Hamiltonian it produces with the one we get from Jordan-Wigner mapping. These two Hamiltonians are expressing the same operator but in different basis. Therefore, their eigenvalues should be equals.

Let's check that this is true. This may take a while.

.. code:: python

    jw_mapping = JordanWigner(num_spin_orbitals)
    pa_jw_qubit_hamiltonien = rd_mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)

    eigvals_jw = np.linalg.eigvals(pa_jw_qubit_hamiltonien.to_matrix())
    eigvals_rd = np.linalg.eigvals(pa_rd_qubit_hamiltonien.to_matrix())

    print(np.all(np.sort(eigvals_jw) == np.sort(eigvals_rd)))

.. code::

    True

.. Add description for BCS Hamiltonian.