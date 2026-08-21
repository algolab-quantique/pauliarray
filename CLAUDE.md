# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the repository root (tests load fixtures via relative paths like `tests/data/integrals/...`).

```bash
flit install --symlink                      # dev install (editable); plain users: pip install .
pip install numpy qiskit qiskit_nature pennylane openfermion black   # full test deps
python -m unittest                          # whole suite (unittest discovery, no pytest)
python -m unittest tests.test_pauli_array   # one module
python -m unittest tests.test_pauli_array.TestPauliArray.test_eq     # one test
black --check --line-length 120 .           # CI fails on any deviation; 120 is not black's default
```

Docs (Sphinx, autodoc + napoleon Google-style docstrings, deployed to GitHub Pages on merge):

```bash
sphinx-apidoc -f -o documentation/source/api_reference .
sphinx-build documentation/source documentation/build
```

CI (`.github/workflows/actions.yml`) runs on pull requests against `main` over Python 3.11 and 3.12: unittest, then the black check. `Self` type hints (PEP 673) are used in `operator_array_type_*.py`, so 3.11 is the floor.

## Architecture

Everything rests on one encoding: an `n`-qubit Pauli string is two length-`n` boolean vectors,

    P = (-i)^(z·x) Z^z X^x

so all Pauli algebra becomes GF(2) linear algebra on `numpy` bool arrays, and array structure comes free from numpy. This is why the library exists — indexing, masking, and broadcasting over Pauli strings are numpy's, not hand-rolled.

**Storage convention.** Every class stores bit strings in arrays with **one extra trailing dimension** of size `num_qubits`: an array of shape `(2, 3)` of 4-qubit Paulis is a `(2, 3, 4)` bool array. `shape` therefore always means `_z_strings.shape[:-1]`. Labels use **little-endian** order — `from_labels("XZZZ")` puts `X` on the highest qubit index — see `label_to_z_string_x_string`, which iterates `reversed(label)`.

**The class stack** (`pauliarray/pauli/`), each layer wrapping the one below:

- `PauliArray` — array of bare Pauli strings (`_z_strings`, `_x_strings`). The base.
- `WeightedPauliArray` — a `PauliArray` plus a complex weight per element; weights and paulis are broadcast to a common shape at construction.
- `Operator` — a *sum*: holds a flattened `WeightedPauliArray`; `num_terms` is its size.
- `OperatorArrayType1` — array of operators sharing a term count; a `WeightedPauliArray` whose **last dimension is the summation axis**.
- `OperatorArrayType2` — array of operators over a **shared Pauli basis**: one `PauliArray` of basis paulis plus a weight matrix whose last axis indexes that basis. Type 1 suits few-term operators of varying content; type 2 suits many operators drawn from one common basis (dense weight matrix, cheap `sum`).

`get_operator(*idx)` / `sum(axis)` move down the stack; `from_operator_list`, `from_pauli_array`, `from_weighted_pauli_array` move up.

**Phases are returned, never stored.** `PauliArray` has no phase field, so `compose_pauli_array` returns `(PauliArray, phases)` — the caller decides where the phase goes (a `WeightedPauliArray` folds it into its weights). Any new `PauliArray` operation that can produce a factor of `±1`/`±i` must follow this two-value-return convention. The `(-i)^(z·x)` normalization is exactly what makes the phase bookkeeping in `compose_pauli_array` a small mod-4 exponent computation.

**GF(2) layer** (`pauliarray/binary/`), where the real algorithms live:

- `bit_operations.py` — `matmul`/`add`/`rank`/`inv`/`row_echelon`/`kernel`/`row_space`/`orthogonal_complement` over bools, mod 2. Generic linear algebra, no Pauli semantics.
- `symplectic.py` — the same on merged `zx_strings` under the symplectic form: commutation (`is_orthogonal`), `is_isotropic`/`is_lagrangian`, `lagrangian_subspace`, `conjugate_subspace`, `gram_schmidt_orthogonalization`. Two Paulis commute iff their zx strings are symplectically orthogonal, so commutation questions become subspace questions here.

Clifford gates (`x`, `h`, `s`, `cx`, `cz`, `clifford_conjugate`) are implemented once per class as bit-string permutations/XORs with phase corrections — they never build matrices. `to_matrix`/`to_matrices` exist for testing and small cases only.

**Fermion mappings** (`pauliarray/mapping/fermion.py`). `FermionMapping` is parameterized entirely by a boolean `mapping_matrix`; `JordanWigner`, `Parity`, and `BravyiKitaev` are three-line subclasses supplying that matrix. Majoranas, creation/annihilation operators, and one-/two-body Hamiltonian assembly are all derived from it, so a new mapping means a new matrix, not new operator code.

**Conversions** (`pauliarray/conversion/`) to qiskit / openfermion / pennylane. These are the only modules importing third-party quantum libraries — the core depends on numpy alone (`pyproject.toml` lists only numpy), and that separation is deliberate: never import qiskit or openfermion from `pauli/`, `binary/`, `mapping/`, or `utils/`.

## Conventions

- Public API is re-exported in `pauliarray/__init__.py` (`Operator`, `OperatorArrayType1`, `OperatorArrayType2`, `PauliArray`, `WeightedPauliArray`); conversion helpers are imported by full path.
- Internal modules are imported aliased and by module, not by symbol: `import pauliarray.pauli.pauli_array as pa`, `... as wpa`, `... as op`, `... as opa`, `from pauliarray.binary import bit_operations as bitops`. Cross-class type hints go under `if TYPE_CHECKING:` to keep the cycles (`PauliArray` ↔ `Operator`) from breaking imports.
- Shape validation goes through `utils/array_operations.py` (`is_broadcastable`, `broadcast_shape`, `is_concatenatable`) plus bare `assert`s; there are no custom exception types.
- Mutating methods take `inplace: bool = True` and return `self` when in place, a new object otherwise — Clifford gates all follow this.
- Module-level `commutator`, `anticommutator`, `concatenate`, `broadcast_to`, `expand_dims`, `swapaxes`, `moveaxis` mirror numpy's free-function style and are defined per class module; a new class is expected to provide its own.
- Floating-point comparisons are threshold-based (`remove_small_weights`, `simplify`, `from_matrix` default `1e-14`/`1e-9`), not exact.
- Google-style docstrings with `Args:`/`Returns:` on public functions — Sphinx napoleon renders them into the published API reference.
