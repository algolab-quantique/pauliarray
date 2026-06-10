# %%

from pauliarray.pauli import phased_pauli_array as ppa

# %%

ppaulis = ppa.PhasedPauliArray.random((2,3),4)

print(ppaulis.inspect())



# %%