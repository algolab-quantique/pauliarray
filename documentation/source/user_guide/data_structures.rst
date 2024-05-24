===============
Data Structures
===============

All the data structures of `PauliArray` rely on the fact that a Pauli string of :math:`n` qubits can be encoded into two vectors of :math:`n` binary components :math:`\mathbf{z}` and :math:`\mathbf{x}` used in the following definition

.. math::

    \hat{P} = (-i)^{\mathbf{z} \cdot \mathbf{x}} \hat{Z}^{\mathbf{z}} \hat{X}^{\mathbf{x}}

where the exponentiation by a vector is to be interpreted as a tensor product

.. math::

    \hat{Z}^{\mathbf{z}} \equiv  \bigotimes_{q=0}^{n-1} \hat{Z}^{z_q}

with :math:`z_q` the :math:`q` th components of the bit string  :math:`\mathbf{z}`. The dot product :math:`\mathbf{z} \cdot \mathbf{x}` counts the number of :math:`\hat{Y}` operators so the factor :math:`(-i)^{\mathbf{z} \cdot \mathbf{x}}` compensates the factors coming from :math:`\hat{Z}\hat{X} = i\hat{Y}`.


----------
PauliArray
----------

The fundamental data structure :code:`PauliArray` represents a :math:`d`-dimension array of Pauli strings. It uses two arrays of the same shape of bit strings :math:`\mathbf{z}` and :math:`\mathbf{x}` to store this information such that each Pauli string in the PauliArray is given by

.. math::

    \hat{P}_{ij\ldots k} = (-i)^{\mathbf{z}_{ij\ldots k} \cdot \mathbf{x}_{ij\ldots k}} \hat{Z}^{\mathbf{z}_{ij\ldots k}} \hat{X}^{\mathbf{x}_{ij\ldots k}}
    .

The :math:`d`-dimension arrays of bit strings :math:`\mathbf{b}_{ij\ldots k}` are stored as :math:`(d+1)`-dimension arrays :math:`\mathsf{b}` using :code:`numpy.ndarray[bool]` where the last hidden dimension is along the length of the Pauli strings and is of size of :math:`n`. The elements of these arrays are related to the bits of the bit string such that

.. math::

    [\mathsf{b}]_{ij\ldots k q} = [\mathbf{b}_{ij\ldots k}]_q

It can be initialized by providing two :code:`numpy.ndarray[bool]` of the same shape.

.. code:: python

    import numpy as np

    from pauliarray import PauliArray

    num_qubits = 4

    z_strings = np.tri(num_qubits, k=-1, dtype=bool)
    x_strings = np.eye(num_qubits, dtype=bool)

    paulis = PauliArray(z_strings, x_strings)

A convenient initialization method using Pauli string labels is also available. This uses the little-endian labelling convention by default.

.. code:: python

    paulis = PauliArray.from_labels(["IIIX", "IIXZ", "IXZZ", "XZZZ"])


Multidimensional :code:`PauliArray` are also supported. The following code creates a :code:`(4, 2)` 4-qubit :code:`PauliArray`.

.. code:: python

    num_qubits = 4

    z_strings = np.zeros((num_qubits, 2, num_qubits), dtype=bool)
    x_strings = np.zeros((num_qubits, 2, num_qubits), dtype=bool)

    z_strings[:, 0, :] = np.tri(num_qubits, k=-1, dtype=bool)
    x_strings[:, 0, :] = np.eye(num_qubits, dtype=bool)

    z_strings[:, 1, :] = np.tri(num_qubits, k=0, dtype=bool)
    x_strings[:, 1, :] = np.eye(num_qubits, dtype=bool)

    paulis = PauliArray(z_strings, x_strings)

This can also be achieved with labels.

.. code:: python

    paulis = PauliArray.from_labels(
        [
            ["IIIX", "IIIY"],
            ["IIXZ", "IIYZ"],
            ["IXZZ", "IYZZ"],
            ["XZZZ", "YZZZ"],
        ]
    )

------------------
WeightedPauliArray
------------------

A :code:`WeightedPauliArray` is obtained by assigning a complex number to each Pauli string in a PauliArray

.. math::

    w_{ij\ldots k} \hat{P}_{ij\ldots k} .

It can be initialized by providing a :code:`PauliArray` and a :code:`numpy.ndarray[complex]`. Both arrays should have the same shape or at leat be broadcastable.

.. code:: python

    from pauliarray import WeightedPauliArray

    num_qubits = 4

    z_strings = np.tri(num_qubits, k=-1, dtype=bool)
    x_strings = np.eye(num_qubits, dtype=bool)

    paulis = PauliArray(z_strings, x_strings)
    weights = np.array([1, 2, 3, 4], dtype=complex)

    wpaulis = WeightedPauliArray(paulis, weights)

Other initialization methods such as :code:`from_labels_and_weights` and :code:`from_z_strings_and_x_strings_and_weights` also exists for convenience.

--------
Operator
--------

Any :math:`n`-qubits operator :math:`\hat{O}` can be decomposed on the basis of Pauli strings of length :math:`n`

.. math::

    \hat{O} = \sum_s w_s \hat{P}_s
    .

Therefore an :code:`Operator` is simply a sum over a one-dimensional :code:`WeightedPauliArray`. It can be initialized by simply providing a one-dimensional :code:`WeightedPauliArray`.

.. code:: python

    from pauliarray import Operator

    operator = Operator(wpaulis)

------------------
OperatorArrayType1
------------------


It is possible to define an array of operators by using a multidimensional :code:`WeightedPauliArray` and assigning its last dimension as the summation axis

.. math::

    \hat{O}_{ij\ldots k} = \sum_s w_{ij\ldots ks} \hat{P}_{ij\ldots ks}
    .

All the operators in this type of operator array have the same number of Pauli strings. 

It can be initialized by providing a :code:`WeightedPauliArray`. The last dimension is associated to the summation.

.. code:: python

    from pauliarray import OperatorArrayType1

    paulis = PauliArray.from_labels(
        [
            ["IIIX", "IIIY"],
            ["IIXZ", "IIYZ"],
            ["IXZZ", "IYZZ"],
            ["XZZZ", "YZZZ"],
        ]
    )
    wpaulis = WeightedPauliArray(paulis, 0.5)

    operators = OperatorArrayType1(wpaulis)

Other initialization methods such as :code:`from_pauli_array` and :code:`from_weighted_pauli_array` allow to specify the summation axis (or axes), while :code:`from_operator_list` and :code:`from_operator_ndarray` can assemble multiple :code:`Operator` into an :code:`OperatorArrayType1`. 



