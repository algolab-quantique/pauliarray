# %%

from pauliarray.pauli import weighted_pauli_array as wpa

# %%

wpaulis = wpa.WeightedPauliArray.random((2,3),4)

print(wpaulis.inspect())



# %%