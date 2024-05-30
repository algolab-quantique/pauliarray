# PauliArray

Library PauliArray provides data structures to easily create and manipulate arrays of Pauli strings and operators expressed as linear combinations of Pauli strings. It uses Numpy to handle the Pauli strings encoding. This confers to PauliArray the same useful Numpy's features such as indexing, masking and broadcasting.

## Installation

Once you have downloaded/cloned the PauliArray repository, you can install from the directory containing the `pyproject.toml` file with the following command.

```
pip install .
```

### To devellop PauliArray

We recommand using [flit](https://flit.pypa.io/en/stable/) to install PauliArray if you plan to devellop it and to use its symbolic link option. Again, from the directory containing the `pyproject.toml` file with the following command.

```
flit install --symlink
```
