from typing import Protocol

import pauliarray.pauli.pauli_array as pa


class HasPaulis(Protocol):
    paulis: pa.PauliArray

    def replace_paulis(self, new_paulis: pa.PauliArray) -> "HasPaulis": ...
