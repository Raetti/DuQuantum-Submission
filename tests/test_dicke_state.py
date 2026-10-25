"""Unit tests for Dicke state generation."""

import numpy as np
import pytest
import cirq
from src.noise_models import sample_benchmark_environment
from src.dicke_state import create_dicke_state, create_dicke_state_adaptive
from src.scoring import test_circuit as verify_circuit, create_dicke_statevector


def test_create_dicke_statevector():
    """Verify ideal statevector calculations."""
    # N=2, n=1 -> (|01> + |10>) / sqrt(2)
    vec = create_dicke_statevector(2, 1)
    expected = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    np.testing.assert_allclose(vec, expected)


@pytest.mark.parametrize("n_qubits, excitations", [
    (2, 1),
    (3, 1),
    (4, 1),
    (3, 2),
    (4, 2),
])
def test_create_dicke_state_fidelity(n_qubits, excitations):
    """Verify circuit fidelity against statevector on benchmark environment."""
    env = sample_benchmark_environment()
    circuit, qubits = create_dicke_state(
        num_qubits=n_qubits,
        num_excitations=excitations,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
        zero_errors=env["zero_errors"],
        one_errors=env["one_errors"],
    )

    assert len(qubits) == n_qubits
    assert len(circuit) <= 1750
    # Verify exact state preparation
    verify_circuit(circuit, excitations, qubits)


def test_create_dicke_state_adaptive():
    """Verify adaptive circuit generation on baseline parameters."""
    circuit, qubits = create_dicke_state_adaptive(num_qubits=3, num_excitations=1)
    assert len(qubits) == 3
    verify_circuit(circuit, 1, qubits)
