import abc
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

import pauliarray.pauli.pauli_array as pa

# import pauliarray.state.basis_state_array as bsa

# from qiskit.primitives import Sampler
# from qiskit.quantum_info import Statevector

# import pauliarray.state.qubit_state as qbs


class BaseEstimator(object):
    """
    Base class for a PauliArray estimator.
    """

    def __init__(self, pauli_object):
        self._pauli_object = pauli_object

        self._last_paulis_expectation_values = None
        self._last_paulis_covariances = None

    @property
    def pauli_object(self):
        return self._pauli_object

    @property
    def paulis(self):
        return self.pauli_object.paulis

    @property
    def last_paulis_expectation_values(self):
        return self._last_paulis_expectation_values

    @property
    def last_paulis_covariances(self):
        return self._last_paulis_covariances

    @abc.abstractmethod
    def estimate_paulis_on_state_circuit(self, state_circuit: QuantumCircuit) -> NDArray:
        """
        Method (to be implemented in subclass) to estimate the expectation value of Paulis in a PauliArray

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            NDArray: _description_
        """
        return

    def update_pauli_object(self, pauli_object):
        """
        Update the weights in the Pauli Object. he underlying PauliArray should be exactly the same.

        Args:
            pauli_object (_type_): _description_
        """
        assert type(self.pauli_object) == type(pauli_object)

        self._pauli_object.update_weights_from_other(pauli_object)

    def estimate_on_state_circuit(self, state_circuit: QuantumCircuit):
        """
        Estimate the expectation value of the Pauli Object given a quantum state stated as a quantum state.

        Args:
            state_circuit (QuantumCircuit): A state given in the form of QuantumCircuit

        Returns:
            _type_: _description_
        """

        paulis_expectation_values, paulis_covariances = self.estimate_paulis_on_state_circuit(state_circuit)

        self._last_paulis_expectation_values = paulis_expectation_values
        self._last_paulis_covariances = paulis_covariances

        pauli_object_expectation_values = self.pauli_object.expectation_values_from_paulis(paulis_expectation_values)
        pauli_object_covariances = self.pauli_object.covariances_from_paulis(paulis_covariances)

        return pauli_object_expectation_values, pauli_object_covariances

    @staticmethod
    def estimate_diagonal_paulis_expectation_values_on_binary_probabilities(
        diagonal_paulis: pa.PauliArray, binary_probabilities: dict
    ):
        """
        Estimate the expectations values of diagonal Pauli Strings given probabilities of measuring different basis states.

        Args:
            diag_paulis (pa.PauliArray): Diagonal Pauli strings
            binary_probabilities (dict): The probabilities associated of measuring basis states.

        Raises:
            ValueError: If the Pauli string are not diagonal.

        Returns:
            NDArray[np.float_]: The expectation values of the Pauli strings. Has the same shape as diag_paulis.
        """
        if not np.all(diagonal_paulis.is_diagonal()):
            raise ValueError("PauliArray provided must contain only diagonal PauliStrings.")

        labels = list(binary_probabilities.keys())
        probabilities = np.array(list(binary_probabilities.values()))

        basis_states = bsa.BasisStateArray.from_labels(labels)

        eigenvalues = basis_states[:, None].diagonal_pauli_array_eigenvalues_values(diagonal_paulis[None, :])
        expectation_values = np.real(np.sum(probabilities[:, None] * eigenvalues, axis=0))

        return expectation_values

    @staticmethod
    def estimate_diagonal_paulis_covariances_on_binary_probabilities(
        diagonal_paulis: pa.PauliArray, binary_probabilities: dict
    ):
        """
        Estimate the covariance matrix of diagonal Pauli Strings given probabilities of measuring different basis states.

        Args:
            diag_paulis (pa.PauliArray): Diagonal Pauli strings
            binary_probabilities (dict): The probabilities associated of measuring basis states.

        Raises:
            ValueError: If the Pauli string are not diagonal.

        Returns:
            NDArray[np.float_]: The expectation values of the Pauli strings. Has the same shape as diag_paulis.
        """
        if not np.all(diagonal_paulis.is_diagonal()):
            raise ValueError("PauliArray provided must contain only diagonal PauliStrings.")

        labels = list(binary_probabilities.keys())
        probabilities = np.array(list(binary_probabilities.values()))

        # basis_states = bsa.BasisStateArray.from_labels(labels)

        # paulis_eigenvalues = basis_states[:, None].diagonal_pauli_array_eigenvalues_values(diagonal_paulis[None, :])
        # paulis_expectation_values = np.real(np.sum(probabilities[:, None] * paulis_eigenvalues, axis=0))

        # prod_paulis, _ = diagonal_paulis[:, None].mul_pauli_array(diagonal_paulis[None, :])

        # prod_paulis_eigenvalues = basis_states[:, None, None].diagonal_pauli_array_eigenvalues_values(
        #     prod_paulis[None, :, :]
        # )
        # prod_paulis_expectation_values = np.real(np.sum(probabilities[:, None, None] * prod_paulis_eigenvalues, axis=0))

        # covariances = (
        #     prod_paulis_expectation_values - paulis_expectation_values[:, None] * paulis_expectation_values[None, :]
        # )

        # return covariances
