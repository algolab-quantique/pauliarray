def transform_paulis_with_clifford(clifford_fct):

    def decorated_clifford_fct(pauli_obj, *args, inplace: bool = False):

        return pauli_obj.clifford_transform_paulis(clifford_fct, *args, inplace=inplace)

    return decorated_clifford_fct
