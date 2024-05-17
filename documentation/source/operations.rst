======================
Operations
======================

All the data structures of PauliArray can be acted on or combined using a set of operations.

--------------------
Indexing and Masking
--------------------

Indexing and masking in PauliArray works similarly as in Numpy. For example, the following code shows how to access the two first Pauli strings of the second column of an PauliArray.

.. code:: python

    from pauliarray import PauliArray

    paulis = PauliArray.from_labels(
        [
            ["IIIX", "IIIY"],
            ["IIXZ", "IIYZ"],
            ["IXZZ", "IYZZ"],
            ["XZZZ", "YZZZ"],
        ]
    )
    new_paulis = paulis[:2, 1]

The result can be seen using the :code:`inspect` method.

.. code:: python

    print(new_paulis.inspect())

.. code:: 

    PauliArray
    IIIY
    IIYZ

-----------
Composition
-----------

The operation of acting on an operator with another operator is called composition. It is equivalent to a matrix product between the matrices representation of the operators. In PauliArray the composition of two arrays is element-wise. For example, the composition of two :code:`WeightedPauliArray` yields a new :code:`WeightedPauliArray`

.. math::

    w_i^{(1)}\hat{P}_i^{(1)} w_j^{(2)} \hat{P}_j^{(2)} = w_{ij}^{(3)} \hat{P}_{ij}^{(3)}.



.. code:: python

    from pauliarray import WeightedPauliArray

    wpaulis_1 = WeightedPauliArray.from_labels_and_weights(["IZ", "ZI"], [1, 2])
    wpaulis_2 = WeightedPauliArray.from_labels_and_weights(["ZZ", "XX"], [3, 4])

    wpaulis_3 = wpaulis_1.compose(wpaulis_2)

    print(wpaulis_3.inspect())

.. code:: 

    PauliArray
    (+3.0000 +0.0000j) ZI
    (+0.0000 +8.0000j) YX

The same two :code:`WeightedPauliArray` can be composed in a outer product fashion such that all the elements from the first :code:`WeightedPauliArray` are composed with all the elements of a second :code:`WeightedPauliArray` 

.. math::

    w_i^{(1)}\hat{P}_i^{(1)} w_j^{(2)} \hat{P}_j^{(2)} = w_{ij}^{(4)} \hat{P}_{ij}^{(4)}.

This results into a 2-dimensionnal :code:`WeightedPauliArray`.

In PauliArray this can be acheived by making use of broadcasting by introducing new dimensions to the arrays with :code:`None`. See `Numpy's documentation <https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools>`_ for more details.

.. code:: python

    wpaulis_4 = wpaulis_1[:, None].compose(wpaulis_2[None, :])

    print(wpaulis_4.inspect())

.. code:: 

    PauliArray
    (+3.0000 +0.0000j) ZI  (+0.0000 +4.0000j) XY
    (+6.0000 +0.0000j) IZ  (+0.0000 +8.0000j) YX


The composition of two :code:`Operator`  :math:`\hat{O}^{(1)} = \sum_{i} w_i^{(1)} \hat{P}_i^{(1)}` and :math:`\hat{O}^{(2)} = \sum_{j} w_j^{(2)} \hat{P}_j^{(2)}` involves such a 2-dimensionnal :code:`WeightedPauliArray`.

.. math::

    \hat{O}^{(1)} \hat{O}^{(2)} = \sum_{i,j} w_i^{(1)} \hat{P}_i^{(1)} w_j^{(2)} \hat{P}_j^{(2)}
    = \sum_{i,j} w_{ij}^{(3)} \hat{P}_{ij}^{(3)}

However, it needs to be flatten (:math:`(i,j) \to k`) to represent an :code:`Operator`.

PauliArray handles compositions of :code:`Operator` this way. It also combines the coefficients of repeated Pauli strings within the sum.

.. math::

    \hat{O}^{(1)} \hat{O}^{(2)} = \sum_k w_{k}^{(3)} \hat{P}_{k}^{(3)}

.. code:: python

    from pauliarray import Operator

    operator_1 = Operator.from_labels_and_weights(["IZ", "XI"], [1, 2])
    operator_2 = Operator.from_labels_and_weights(["II", "XZ"], [2, 1])

    operator_3 = operator_1.compose(operator_2)

    print(operator_3.inspect())

.. code:: 

    Operator Sum of
    (+5.0000 +0.0000j) XI
    (+4.0000 +0.0000j) IZ



-----------
Commutation
-----------

Based on the encoding of Pauli strings with bit strings :math:`\mathbf{z}` and :math:`\mathbf{x}`, it's easy to show that Pauli strings :math:`\hat{P}^{(1)}` and :math:`\hat{P}^{(2)}` commute if

.. math::
    :name: do_commute

    c = \mathbf{z}^{(1)} \cdot \mathbf{x}^{(2)} + \mathbf{x}^{(1)} \cdot \mathbf{z}^{(2)} \pmod{2}

is equal to :math:`0` and anticommute otherwise. Commutation can be assessed element-wise using the :code:`commute_with` method.

.. code:: python

    from pauliarray import PauliArray

    paulis_1 = PauliArray.from_labels(["IZ", "ZI"])
    paulis_2 = PauliArray.from_labels(["ZZ", "XX"])

    do_commute = paulis_1.commute_with(paulis_2)

    print(do_commute)

.. code::

    [ True False]

Actual commutator can be computed element-wise between two arrays of Pauli strings

.. math::

    [\hat{P}^{(1)}_i, \hat{P}^{(2)}_i] = \hat{P}^{(1)}_i \hat{P}^{(2)}_i - \hat{P}^{(2)}_i \hat{P}^{(1)}_i
    .
    

For efficiency, this operation can be reduced to a single composition

.. math::
    
    [\hat{P}^{(1)}_i, \hat{P}^{(2)}_i] = 2c_i \hat{P}^{(1)}_i \hat{P}^{(2)}_i
    
where :math:`c_i` is given by the equation :eq:`do_commute`.



.. is $1$ if $[\hat{P}^{(1)}_i, \hat{P}^{(2)}_j] \neq 0$.