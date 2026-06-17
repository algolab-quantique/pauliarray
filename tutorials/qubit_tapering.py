# %%

from qiskit_nature.second_q.drivers import PySCFDriver

import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.conversion.qiskit import extract_fermionic_op, weighted_pauli_array_from_pauli_list
from pauliarray.diagonalisation.commutating_paulis.with_circuits import (
    general_to_diagonal as general_to_diagonal_with_circuit,
)
from pauliarray.diagonalisation.commutating_paulis.with_operators import (
    general_to_diagonal as general_to_diagonal_with_operators,
)
from pauliarray.diagonalisation.commutating_paulis.with_qiskit_circuits import (
    general_to_diagonal as general_to_diagonal_with_qiskit_circuits,
)
from pauliarray.factorisation import qubit_tapering as qutap
from pauliarray.mapping.fermion import BravyiKitaev, JordanWigner, Parity

# %%

mol_info = {
    # "atom": "N 0 0 -0.545;N 0 0 0.545;",
    # "atom": "Li 0 0 0;H 0 0 1.6;",
    "atom": "H 0 0 0;H 0 0 0.735;",
    "basis": "sto3g",
    "charge": 0,
    "spin": 0,
}

driver = PySCFDriver(**mol_info)
problem = driver.run()
hamiltonian = problem.hamiltonian
second_q_hamiltonian = problem.hamiltonian.second_q_op()

num_spin_orbitals = second_q_hamiltonian.num_spin_orbitals

one_body_tuple, two_body_tuple = extract_fermionic_op(second_q_hamiltonian)

mapping = JordanWigner(num_spin_orbitals)
qubit_hamiltonian = mapping.assemble_qubit_hamiltonian_from_sparses(one_body_tuple, two_body_tuple)


print(qubit_hamiltonian.inspect())
# %%

sym_paulis = qutap.find_symmetry_paulis(qubit_hamiltonian.paulis)
print(sym_paulis.inspect())

transformations = qutap.symmetries_to_qubit_transformations(sym_paulis)
print(transformations.inspect())

transformed_qubit_hamiltonian: op.Operator = transformations.successive_clifford_conjugate_pauli_obj(qubit_hamiltonian)
print(transformed_qubit_hamiltonian.inspect())

# %%
