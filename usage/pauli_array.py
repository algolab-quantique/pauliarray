# %%

from pauliarray.pauli import pauli_array as pa

# %%

paulis = pa.PauliArray.random((2,3),4)

print(paulis.inspect())

# %%