from typing import Protocol

import pauliarray.pauli.pauli_array as pa


class HasPaulis(Protocol):
    paulis: pa.PauliArray
