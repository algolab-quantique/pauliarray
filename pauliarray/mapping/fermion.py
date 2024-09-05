import itertools
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import pauliarray.pauli.operator as op
import pauliarray.pauli.operator_array_type_1 as opa
import pauliarray.pauli.pauli_array as pa
import pauliarray.pauli.weighted_pauli_array as wpa
from pauliarray.binary import bit_operations as bitops


class FermionMapping(object):
    """
    Base class to represent a Fermion-to-qubit mapping.

    Attributes:
        mapping_matrix ("np.ndarray[np.bool]"): A boolean numpy array representing the mapping matrix.
        name (str): A name identifier for the mapping (default is "mapping").
        _mapping_matrix_inv ("np.ndarray[np.bool]", optional): The inverse of the mapping matrix, initially None
        and computed on demand.
        _heavyside_matrix ("np.ndarray[np.bool]", optional): The Heavyside matrix, initially None and computed
        on demand.
        _parity_matrix ("np.ndarray[np.bool]", optional): The parity matrix, initially None and computed on demand.
    """

    def __init__(self, mapping_matrix: "np.ndarray[np.bool]", name: str = "mapping"):
        """
        Constructs all the necessary attributes for the FermionMapping object.

        Args:
            mapping_matrix : "np.ndarray[np.bool]"
                A boolean numpy array representing the mapping matrix.
            name : str, optional
                A name identifier for the mapping (default is "mapping").
        """
        self.mapping_matrix = mapping_matrix
        self.name = name

        self._mapping_matrix_inv = None
        self._heavyside_matrix = None
        self._parity_matrix = None

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits based on the shape of the mapping matrix.

        Returns:
            int: The number of qubits.
        """
        return self.mapping_matrix.shape[0]

    @property
    def mapping_matrix_inv(self):
        """
        Returns the inverse of the mapping matrix. Computes it if not already computed.

        Returns:
            "np.ndarray[np.bool]": The inverse of the mapping matrix.
        """
        if self._mapping_matrix_inv is None:
            self._mapping_matrix_inv = bitops.inv(self.mapping_matrix)

        return self._mapping_matrix_inv

    @property
    def heavyside_matrix(self) -> "np.ndarray[np.bool]":
        """
        Returns the Heavyside matrix. Computes it if not already computed.

        Returns:
            "np.ndarray[np.bool]": The Heavyside matrix.
        """
        if self._heavyside_matrix is None:
            self._heavyside_matrix = np.tri(self.num_qubits, k=-1, dtype=np.bool_)

        return self._heavyside_matrix

    @property
    def parity_matrix(self) -> "np.ndarray[np.bool]":
        """
        Returns the parity matrix. Computes it if not already computed.

        Returns:
            "np.ndarray[np.bool]": The parity matrix.
        """
        if self._parity_matrix is None:
            self._parity_matrix = np.tri(self.num_qubits, dtype=np.bool_)

        return self._parity_matrix

    def majoranas(self) -> Tuple[pa.PauliArray, pa.PauliArray]:
        r"""
        In a fermion-to-qubit mapping, each creation/annihilation operator is a sum of two majorana operators,

        .. math::

            0.5 * (P_\text{real} + P_\text{imag})

        each being a Pauli string. This methods construct these majorana operators.

        Returns:
            PauliArray: The Pauli strings for :math:`P_\text{real}`
            PauliArray: The Pauli strings for :math:`P_\text{imag}`
        """

        mapping_matrix = self.mapping_matrix
        mapping_matrix_inv = self.mapping_matrix_inv

        heavyside_matrix = self.heavyside_matrix
        parity_matrix = self.parity_matrix

        real_z_strings = bitops.matmul(heavyside_matrix, mapping_matrix_inv)
        imag_z_strings = bitops.matmul(parity_matrix, mapping_matrix_inv)
        real_x_strings = imag_x_strings = mapping_matrix.transpose()

        real_majoranas = pa.PauliArray.from_z_strings_and_x_strings(real_z_strings, real_x_strings)
        imag_majoranas = pa.PauliArray.from_z_strings_and_x_strings(imag_z_strings, imag_x_strings)

        return real_majoranas, imag_majoranas

    def assemble_creation_annihilation_operators(self) -> Tuple[opa.OperatorArrayType1, opa.OperatorArrayType1]:
        """
        Constructs the creation and annihilation operators for all available states and returns them as OperatorArrays.

        Returns:
            OperatorArrayType1: The creation operators
            OperatorArrayType1: The annihilation operators
        """
        real_majoranas, imag_majoranas = self.majoranas()

        real_operators = opa.OperatorArrayType1.from_pauli_array(real_majoranas)
        imag_operators = opa.OperatorArrayType1.from_pauli_array(imag_majoranas)

        annihilation_operators = 0.5 * real_operators.add_operator_array_type_1(1j * imag_operators)
        creation_operators = annihilation_operators.adjoint()

        return creation_operators, annihilation_operators

    def occupation_operators(self) -> opa.OperatorArrayType1:
        """
        Constructs the occupation operators for all available states and returns them as OperatorArray.

        Returns:
            opa.OperatorArrayType1: An Operator array with all the occupation operators.
        """
        mapping_matrix_inv = self.mapping_matrix_inv

        identity_paulis = pa.PauliArray.from_z_strings_and_x_strings(
            np.zeros((self.num_qubits, self.num_qubits), dtype=np.bool_),
            np.zeros((self.num_qubits, self.num_qubits), dtype=np.bool_),
        )
        z_paulis = pa.PauliArray.from_z_strings_and_x_strings(
            mapping_matrix_inv, np.zeros((self.num_qubits, self.num_qubits), dtype=np.bool_)
        )

        identity_operators = opa.OperatorArrayType1.from_pauli_array(identity_paulis)
        z_operators = opa.OperatorArrayType1.from_pauli_array(z_paulis)

        occupation_operators = 0.5 * identity_operators.add_operator_array_type_1(-1 * z_operators)

        return occupation_operators

    def assemble_qubit_hamiltonian_from_arrays(
        self, one_body: "np.ndarray[np.complex]", two_body: "np.ndarray[np.complex]"
    ) -> op.Operator:
        """
        Assemble the whole qubit Hamiltonian as an Operator using fermionic integrals given as arrays.

        Args:
            one_body ("np.ndarray[np.complex]"): The one-body fermionic integrals as a 2d array
            two_body ("np.ndarray[np.complex]"): The two-body fermionic integrals as a 4d array (in physicist order)

        Returns:
            Operator: The qubit Hamiltonian
        """
        one_body_operator = self.one_body_operator_from_array(one_body)
        two_body_operator = self.two_body_operator_from_array(two_body)

        return (one_body_operator + two_body_operator).combine_repeated_terms().remove_small_weights()

    def assemble_qubit_hamiltonian_from_sparses(self, one_body_tuple: Tuple, two_body_tuple: Tuple) -> op.Operator:
        """
        Assemble the whole qubit Hamiltonian as an Operator using fermionic integrals given in sparse representations.

        Args:
            one_body_tuple (Tuple): Contains the argument for the `one_body_operator_from_sparse` method.
            two_body_tuple (Tuple): Contains the argument for the `two_body_operator_from_sparse` method.

        Returns:
            op.Operator: The qubit Hamiltonian
        """
        one_body_operator = self.one_body_operator_from_sparse(*one_body_tuple)
        two_body_operator = self.two_body_operator_from_sparse(*two_body_tuple)

        return (one_body_operator + two_body_operator).combine_repeated_terms().remove_small_weights()

    def one_body_operator_from_sparse(
        self,
        locations: List["np.ndarray[np.int]"],
        values: "np.ndarray[np.complex]",
        signs: Union[List[int], "np.ndarray[np.int]", List["np.ndarray[np.int]"]] = (1, -1),
    ) -> op.Operator:
        """
        Assemble a one body fermionic operator as an Operator using fermionic integrals given in sparse representations.

        Args:
            locations (List["np.ndarray[np.int]"]): Pairs of orbital indices
            values ("np.ndarray[np.complex]"): Values of the integral for the pairs of orbitals
            signs (List[int] or "np.ndarray[np.int]" or List["np.ndarray[np.int]"]): Values +1 or -1 determining if
            the operators are creation or annihilation. Can be a list of two signs, or two arrays of sign.
            Defaults to [1,-1].

        Returns:
            op.Operator: The qubit operator
        """

        num_terms = len(values)
        assert len(locations) == 2
        for index in range(2):
            assert len(locations[index]) == num_terms

        assert len(signs) == 2
        signs = list(signs)
        for index in range(2):
            if isinstance(signs[index], np.ndarray):
                assert len(signs[index]) == num_terms
            else:
                signs[index] = signs[index] * np.ones(num_terms)

        _is, _js = locations
        isigns, jsigns = signs

        flip_ij_operators = self._flip_operators(_is, isigns * self._flip_factors(_is, _js))
        flip_j_operators = self._flip_operators(_js, jsigns)

        update_operators = self._update_operators(_is, _js)

        all_operators = (
            update_operators.compose_operator_array_type_1(flip_ij_operators)
            .compose_operator_array_type_1(flip_j_operators)
            .mul_weights(values)
        )

        one_body_operator = all_operators.sum()

        return one_body_operator

    def two_body_operator_from_sparse(
        self,
        locations: List["np.ndarray[np.int]"],
        values: "np.ndarray[np.complex]",
        signs: Union[List[int], "np.ndarray[np.int]", List["np.ndarray[np.int]"]] = (1, 1, -1, -1),
    ) -> op.Operator:
        """
        Assemble a two-body fermionic operator as an Operator using fermionic integrals given in sparse representations.

        Args:
            locations (List["np.ndarray[np.int]"]): Quartet of orbital indices
            values ("np.ndarray[np.complex]"): Values of the integral for the quartets of orbitals
            signs (List[int] or "np.ndarray[np.int]" or List["np.ndarray[np.int]"]): Values +1 or -1 determining if
            the operators are creation or annihilation. Can be a list of two signs, or two arrays of sign.
            Defaults to [+1,+1,-1,-1].

        Returns:
            op.Operator: The qubit operator
        """

        num_terms = len(values)
        assert len(locations) == 4
        for index in range(4):
            assert len(locations[index]) == num_terms

        assert len(signs) == 4
        signs = list(signs)
        for index in range(4):
            if isinstance(signs[index], np.ndarray):
                assert len(signs[index]) == num_terms
            else:
                signs[index] = signs[index] * np.ones(num_terms)

        _is, _js, _ks, _ls = locations

        isigns, jsigns, ksigns, lsigns = signs

        flip_ijkl_operators = self._flip_operators(_is, isigns * self._flip_factors(_is, _js, _ks, _ls))
        flip_jkl_operators = self._flip_operators(_js, jsigns * self._flip_factors(_js, _ks, _ls))
        flip_kl_operators = self._flip_operators(_ks, ksigns * self._flip_factors(_ks, _ls))
        flip_l_operators = self._flip_operators(_ls, lsigns)

        update_operators = self._update_operators(_is, _js, _ks, _ls)

        all_operators = (
            update_operators.compose_operator_array_type_1(flip_ijkl_operators)
            .compose_operator_array_type_1(flip_jkl_operators)
            .compose_operator_array_type_1(flip_kl_operators)
            .compose_operator_array_type_1(flip_l_operators)
        ).mul_weights(values)

        two_body_operator = all_operators.sum()

        return two_body_operator

    def one_body_operator_from_array(self, one_body: "np.ndarray[np.complex]") -> op.Operator:
        """
        Converts a one-body array to an operator object.

        Args:
            one_body ("np.ndarray[np.complex]"): A complex numpy array representing the one-body operator.

        Returns:
            op.Operator: The corresponding operator object.

        Raises:
            AssertionError. If the input array is not 2-dimensional or if its dimensions do not match the
            number of qubits.
        """
        assert one_body.ndim == 2
        assert all([s == self.num_qubits for s in one_body.shape])

        non_zero_coef = ~np.isclose(one_body, 0)

        locations = np.where(non_zero_coef)
        values = one_body[locations[0], locations[1]]

        return self.one_body_operator_from_sparse(locations, values)

    def two_body_operator_from_array(self, two_body: "np.ndarray[np.complex]") -> op.Operator:
        """
        Converts a two-body array to an operator object.

        Parameters:
            two_body ("np.ndarray[np.complex]"): A complex numpy array representing the two-body operator.

        Returns:
            op.Operator: The corresponding operator object.

        Raises:
            AssertionError. If the input array's dimensions do not match the number of qubits or if it does not satisfy
            the required symmetries.
        """
        assert all([s == self.num_qubits for s in two_body.shape])

        assert np.all(np.isclose(two_body, np.einsum("ijkl->ikjl", two_body)))
        assert np.all(np.isclose(two_body, np.einsum("ijkl->ljki", two_body)))
        assert np.all(np.isclose(two_body, np.einsum("ijkl->jilk", two_body)))

        non_zero_coef = ~np.isclose(two_body, 0)

        locations = np.where(non_zero_coef)
        values = two_body[locations[0], locations[1], locations[2], locations[3]]

        return self.two_body_operator_from_sparse(locations, values)

    def assemble_one_body_operator_array(self) -> opa.OperatorArrayType1:
        """
        Assembles an array of one-body operators.

        Returns:
            opa.OperatorArrayType1: An operator array of one-body operators.
        """
        creation_operators, annihilation_operators = self.assemble_creation_annihilation_operators()

        return creation_operators[:, None].compose_operator_array_type_1(annihilation_operators[None, :])

    def assemble_two_body_operator_array(self) -> opa.OperatorArrayType1:
        """
        Assembles an array of two-body operators.

        Returns:
            opa.OperatorArrayType1: An operator array of two-body operators.
        """
        creation_operators, annihilation_operators = self.assemble_creation_annihilation_operators()
        double_annihilation = annihilation_operators[:, None].compose_operator_array_type_1(
            annihilation_operators[None, :]
        )
        double_creaation = creation_operators[:, None].compose_operator_array_type_1(creation_operators[None, :])

        tmp = double_creaation[:, :, None, None].compose_operator_array_type_1(double_annihilation[None, None, :, :])

        return tmp

    def _flip_operators(self, i_orbitals: "np.ndarray[np.int]", factors: NDArray[np.float64]) -> opa.OperatorArrayType1:
        r"""
        Constructs an OperatorArray with the :math:`\mu^\text{th}` flip operators acting on the orbitals :math:`i_\mu`

        .. math::
            \hat{F}_\mu = \frac{1}{2}(1 + f_\mu \hat{Z}_q^{[\mathsf{M}^{-1}]_{i_\mu q}]})

        where :math:`f_\mu` is a factor (+1 or -1) associted with the creation and annihilation operators.

        Args:
            i_orbitals (NDArray[int]): The indices of the orbital i the operator is acting on
            factors (NDArray[float]): The factors (+1 or -1) defining if a creation or an annihilation operator is
            applied.

        Returns:
            OperatorArrayType1: The array of flip operators.
        """

        z_strings = self.mapping_matrix_inv[i_orbitals, :]

        z_operators = opa.OperatorArrayType1.from_pauli_array(
            pa.PauliArray.from_z_strings_and_x_strings(z_strings, np.zeros(z_strings.shape, dtype=np.bool_))
        )

        return z_operators.mul_weights(0.5 * factors).add_scalar(0.5)

    def _update_operators(
        self, i_orbitals: "np.ndarray[np.int]", j_orbitals: "np.ndarray[np.int]", *args: Tuple["np.ndarray[np.int]"]
    ) -> opa.OperatorArrayType1:
        """
        Updates the operators based on the given orbitals, using the
        Heaviside matrix, the inverse mapping matrix, and the transposed mapping matrix.
        The function processes the input orbitals to generate updated Z and X strings,
        used to create new Pauli operators.

        Args:
            i_orbitals ("np.ndarray[np.int]"): The first set of orbitals.
            j_orbitals ("np.ndarray[np.int]"): The second set of orbitals.
            *args (Tuple["np.ndarray[np.int]"]): Additional sets of orbitals.

        Returns:
            opa.OperatorArrayType1: The updated operators as an OperatorArrayType1 object.
        """
        heavy_map_inv = bitops.matmul(self.heavyside_matrix, self.mapping_matrix_inv).astype(np.int8)
        mapping_matrix_tra = self.mapping_matrix.transpose().astype(np.int8)

        xs_orbitals = (
            i_orbitals,
            j_orbitals,
        ) + args

        update_z_strings = 0
        update_x_strings = 0
        for x_orbitals in xs_orbitals:
            update_z_strings += heavy_map_inv[x_orbitals, :]
            update_x_strings += mapping_matrix_tra[x_orbitals, :]

        update_z_strings = np.mod(update_z_strings, 2).astype(np.bool_)
        update_x_strings = np.mod(update_x_strings, 2).astype(np.bool_)

        update_factors = self._update_factors(*xs_orbitals)

        update_paulis = pa.PauliArray.from_z_strings_and_x_strings(update_z_strings, update_x_strings)
        update_operators = opa.OperatorArrayType1.from_pauli_array(update_paulis).mul_weights(update_factors)

        return update_operators

    def _update_operators_2(
        self, i_orbitals: "np.ndarray[np.int]", j_orbitals: "np.ndarray[np.int]"
    ) -> opa.OperatorArrayType1:
        r"""
        Constructs an OperatorArray with the :math:`\mu^\text{th}` main operators acting on the orbitals :math:`i_\mu`
        and :math:`j_\mu` in a one-body fermionic operator

        .. math::

            \hat{U}^{(2)}_{\mu} = (-1)^{\theta_{i_\mu j_ \mu}} \hat{X}_q^{M_{qi_\mu} + M_{qj_\mu}}.


        Args:
            i_orbitals (NDArray[int]): The indices of the orbital i the operator is acting on.
            j_orbitals (NDArray[int]): The indices of the orbital j the operator is acting on.

        Returns:
            OperatorArrayType1: The array of operators.
        """

        heavy_map_inv = bitops.matmul(self.heavyside_matrix, self.mapping_matrix_inv).astype(np.int8)
        mapping_matrix_tra = self.mapping_matrix.transpose().astype(np.int8)

        update_z_strings = bitops.add(heavy_map_inv[i_orbitals, :], heavy_map_inv[j_orbitals, :])
        update_x_strings = bitops.add(mapping_matrix_tra[i_orbitals, :], mapping_matrix_tra[j_orbitals, :])

        update_factors = self._update_factors(i_orbitals, j_orbitals)

        update_paulis = pa.PauliArray.from_z_strings_and_x_strings(update_z_strings, update_x_strings)
        update_operators = opa.OperatorArrayType1.from_pauli_array(update_paulis).mul_weights(update_factors)

        return update_operators

    def _update_operators_4(
        self,
        i_orbitals: "np.ndarray[np.int]",
        j_orbitals: "np.ndarray[np.int]",
        k_orbitals: "np.ndarray[np.int]",
        l_orbitals: "np.ndarray[np.int]",
    ) -> opa.OperatorArrayType1:
        r"""
        Constructs an OperatorArray with the :math:`\mu^\text{th}` main operators acting on the orbitals
        :math:`i_\mu`, :math:`j_\mu`, :math:`k_\mu` and :math:`l_\mu` in a two-body fermionic operator.

        .. math::

            (-1)^{\theta_{ij} + \theta_{i_\mu k_\mu} + \theta_{i_\mu l_\mu} + \theta_{j_\mu k_\mu}  +
                \theta_{j_\mu l_\mu} + \theta_{k_\mu l_\mu}}
                \hat{X}_q^{M_{qi_\mu} + M_{qj_\mu} + M_{qk_\mu} + M_{ql_\mu}}
                \hat{Z}_q^{(\theta_{i_\mu p} + \theta_{j_\mu p} + \theta_{k_\mu p} +
                \theta_{l_\mu p})[\mathsf{M}^{-1}]_{pq}}


        Args:
            i_orbitals (NDArray[int]): The indices of the orbital i the operator is acting on.
            j_orbitals (NDArray[int]): The indices of the orbital j the operator is acting on.
            k_orbitals (NDArray[int]): The indices of the orbital k the operator is acting on.
            l_orbitals (NDArray[int]): The indices of the orbital l the operator is acting on.

        Returns:
            OperatorArrayType1: The array of operators.
        """

        heavy_map_inv = bitops.matmul(self.heavyside_matrix, self.mapping_matrix_inv).astype(np.int8)
        mapping_matrix_tra = self.mapping_matrix.transpose().astype(np.int8)

        update_z_strings = np.mod(
            heavy_map_inv[i_orbitals, :]
            + heavy_map_inv[j_orbitals, :]
            + heavy_map_inv[k_orbitals, :]
            + heavy_map_inv[l_orbitals, :],
            2,
        ).astype(np.bool_)
        update_x_strings = np.mod(
            mapping_matrix_tra[i_orbitals, :]
            + mapping_matrix_tra[j_orbitals, :]
            + mapping_matrix_tra[k_orbitals, :]
            + mapping_matrix_tra[l_orbitals, :],
            2,
        ).astype(np.bool_)

        update_factors = self._update_factors(i_orbitals, j_orbitals, k_orbitals, l_orbitals)

        update_paulis = pa.PauliArray.from_z_strings_and_x_strings(update_z_strings, update_x_strings)
        update_operators = opa.OperatorArrayType1.from_pauli_array(update_paulis).mul_weights(update_factors)

        return update_operators

    @staticmethod
    def _flip_factors(
        i_orbitals: "np.ndarray[np.int]", j_orbitals: "np.ndarray[np.int]", *args: Tuple["np.ndarray[np.int]"]
    ) -> NDArray[np.float64]:
        r"""
        Computes flip factors of the type

        .. math::
            f_\mu = (-1)^{\delta_{i_\mu j_\mu} + \delta_{i_\mu k_\mu} + \ldots}.


        Args:
            i_orbitals ("np.ndarray[np.int]"): The indices of the orbital i the operator is acting on.
            j_orbitals ("np.ndarray[np.int]"): The indices of the orbital i the operator is acting on.
            etc...

        Returns:
            NDArray[float]: The factors.
        """

        xs_orbitals = (j_orbitals,) + args

        factors = 1
        for x_orbitals in xs_orbitals:
            factors *= np.choose(i_orbitals == x_orbitals, [1, -1])

        return factors

    def _update_factors(
        self, i_orbitals: "np.ndarray[np.int]", j_orbitals: "np.ndarray[np.int]", *args: Tuple["np.ndarray[np.int]"]
    ) -> NDArray[np.float64]:
        r"""
        Computes update factors of the type

        .. math::
            f_\mu = (-1)^{\theta_{i_\mu j_\mu} - \theta_{j_\mu i_\mu} + \ldots}.


        Args:
            i_orbitals ("np.ndarray[np.int]"): The indices of the orbital i the operator is acting on.
            j_orbitals ("np.ndarray[np.int]"): The indices of the orbital i the operator is acting on.
            etc...

        Returns:
            NDArray[float]: The factors.
        """

        xs_orbitals = (
            i_orbitals,
            j_orbitals,
        ) + args

        heavyside_matrix = self.heavyside_matrix.astype(np.int8)

        phases = 0
        for x_orbitals, y_orbitals in itertools.combinations(xs_orbitals, 2):
            phases += heavyside_matrix[x_orbitals, y_orbitals] - heavyside_matrix[y_orbitals, x_orbitals]

        # TODO : check which one is faster
        # factors = np.choose(np.mod(phase, 4), [1, 1j, -1, -1j])
        # factors = np.choose(np.bitwise_and(phase - phase, 3), [1, 1j, -1, -1j])
        # factors = np.choose(phase - phase, [1, 1j, -1, -1j] * 2, mode="wrap")

        factors = np.choose(phases, [1, 1j, -1, -1j] * 2, mode="wrap")

        return factors


class JordanWigner(FermionMapping):
    def __init__(self, num_qubits: int):
        """
        Initializes the Jordan-Wigner mapping for a given number of qubits. Maps fermionic operators to qubit operators.

        Args:
            num_qubits (int): The number of qubits to be used in the mapping.
        """
        FermionMapping.__init__(self, np.eye(num_qubits, dtype=np.bool_), "jordan-wigner")

    @property
    def __name__(self):
        """
        Gets the name of the mapping.

        Returns:
            str: The name of the mapping, "JordanWigner".
        """
        return "JordanWigner"


class Parity(FermionMapping):
    def __init__(self, num_qubits: int):
        """
        Initializes the Parity mapping for a given number of qubits. Maps fermionic operators to qubit operators.

        Args:
            num_qubits (int): The number of qubits to be used in the mapping.
        """
        FermionMapping.__init__(self, np.tri(num_qubits, dtype=np.bool_), "parity")

    @property
    def __name__(self):
        """
        Gets the name of the mapping.

        Returns:
            str: The name of the mapping, "Parity".
        """
        return "Parity"


class BravyiKitaev(FermionMapping):
    def __init__(self, num_qubits: int):
        """
        Initialize the Bravyi-Kitaev mapping for a given number of qubits. Maps fermionic operators to qubit operators.

        Args:
            num_qubits (int): The number of qubits to be used in the mapping.
        """
        FermionMapping.__init__(self, self._build_bravyi_kitaev_mapping_matrix(num_qubits), "bravyi-kitaev")

    def _build_bravyi_kitaev_mapping_matrix(self, num_qubits: int) -> "np.ndarray[np.bool]":
        """
        Constructs the Bravyi-Kitaev mapping matrix for the given number of qubits.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            "np.ndarray[np.bool]": The Bravyi-Kitaev mapping matrix.
        """
        mapping_matrix = np.eye(num_qubits, dtype=np.bool_)

        for i in range(1, num_qubits + 1, 2):
            if np.log2(i + 1) % 1 == 0:
                mapping_matrix[i, : i + 1] = True
            else:
                mapping_matrix[i, 2 ** int(np.log2(i + 1)) : i + 1] = True

        return mapping_matrix

    @property
    def __name__(self):
        """
        Gets the name of the mapping.

        Returns:
            str: The name of the mapping, "BravyiKitaev".
        """
        return "BravyiKitaev"
