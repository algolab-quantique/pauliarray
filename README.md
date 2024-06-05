# PauliArray

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

Library PauliArray provides data structures to easily create and manipulate arrays of Pauli strings and operators expressed as linear combinations of Pauli strings. It uses Numpy to handle the Pauli strings encoding. This confers to PauliArray the same useful Numpy's features such as indexing, masking and broadcasting.

## Installation

Once you have downloaded/cloned the PauliArray repository, you can install from the directory containing the `pyproject.toml` file with the following command

```
pip install .
```

### Contribute to PauliArray

To install PauliArray with the aim of editing or developping features, we recommand using [flit](https://flit.pypa.io/en/stable/) for the installation and to use its symbolic link option. Again, from the directory containing the `pyproject.toml` file, execute the following command

```
flit install --symlink
```

## Documentation

Documentation for PauliArray, including tutorials and API reference can be found [here](https://algolab-quantique.github.io/pauliarray/).