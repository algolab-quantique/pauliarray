======================
PauliArray
======================

Welcome to PauliArray's documentation.

.. important::

    This documentation and library are under development. If you want to open an issue or contribute, feel free to do so through the `PauliArray repository <https://github.com/algolab-quantique/pauliarray>`_

This library has been developped as an **efficient manipulation toolbox** for Pauli operators.

.. seealso::
    An indepth description and advantages showcase of PauliArray can be found in the PauliArray technical paper
    available on the `ArXiv <https://arxiv.org/abs/2405.19287>`_.

Installation
=============

Once you have downloaded/cloned the PauliArray repository, you can install from the directory containing the `pyproject.toml` file with the following command

.. code::

    pip install .

To develop PauliArray
----------------------

To install PauliArray with the aim of editing or developping features, we recommand using [flit](https://flit.pypa.io/en/stable/) for the installation and to use its symbolic link option. Again, from the directory containing the `pyproject.toml` file, execute the following command

.. code::

    flit install --symlink


User Guides
=============

The following topics aim to help you understand how to use PauliArray.

.. toctree::
    :maxdepth: 1
    :caption: User Guides
    :glob:

    user_guide/*

::::

Tutorials
=============

The tutorials demonstrate use cases of the PauliArray library.

.. toctree::
    :maxdepth: 1
    :caption: Tutorials
    :glob:

    tutorials/*

::::


API Reference
=============

The following pages contain PauliArray's API reference.


.. toctree::
    :maxdepth: 4
    :caption: API reference

    api_reference/pauliarray