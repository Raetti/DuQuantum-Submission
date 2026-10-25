"""Unit tests for Dicke state generation and optimized grid selection."""

import numpy as np
import pytest
import cirq
from src.noise_models import sample_benchmark_environment
from src.dicke_state import create_dicke_state, create_dicke_state_adaptive
from src.scoring import test_circuit as verify_circuit, create_dicke_statevector
from src.willow_utils import build_connectivity_graph, find_best_linear_chain


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


def test_grid_selection_avoids_high_readout_error():
    """Verify that grid selection steers away from qubits with high readout error."""
    # Path A: (0, 0) - (0, 1) has tiny CZ error (0.001) but terrible readout error (0.15)
    # Path B: (1, 0) - (1, 1) has slightly higher CZ error (0.003) but excellent readout error (0.002)
    qA1, qA2 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    qB1, qB2 = cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)

    cz_errors = {
        (qA1, qA2): (0.001, 0.0),
        (qB1, qB2): (0.003, 0.0),
    }
    sq_errors = {q: (0.0004, 0.0) for q in (qA1, qA2, qB1, qB2)}
    zero_errors = {
        qA1: (0.15, 0.0), qA2: (0.15, 0.0),
        qB1: (0.002, 0.0), qB2: (0.002, 0.0),
    }
    one_errors = {
        qA1: (0.20, 0.0), qA2: (0.20, 0.0),
        qB1: (0.003, 0.0), qB2: (0.003, 0.0),
    }

    graph = build_connectivity_graph(
        cz_errors,
        sq_pauli_error_rate=sq_errors,
        zero_errors=zero_errors,
        one_errors=one_errors,
        num_qubits=2,
        num_excitations=1,
    )
    selected_chain = find_best_linear_chain(graph, length=2, num_excitations=1)
    # Must choose Path B because the readout error penalty far outweighs the minor CZ difference
    assert set(selected_chain) == {qB1, qB2}


def test_grid_selection_orientation_places_stable_qubit_first():
    """Verify that orientation places the more stable endpoint at q0 (which idles longest)."""
    # 3-qubit linear path: q0 - q1 - q2
    q0, q1, q2 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)
    cz_errors = {
        (q0, q1): (0.002, 0.0),
        (q1, q2): (0.002, 0.0),
    }
    # q0 is very noisy / decaying (high relaxation and dephasing)
    # q2 is very stable / quiet
    sq_errors = {
        q0: (0.01, 0.0),
        q1: (0.001, 0.0),
        q2: (0.0001, 0.0),
    }
    one_errors = {
        q0: (0.05, 0.0),
        q1: (0.005, 0.0),
        q2: (0.001, 0.0),
    }

    graph = build_connectivity_graph(
        cz_errors,
        sq_pauli_error_rate=sq_errors,
        one_errors=one_errors,
        num_qubits=3,
        num_excitations=1,
    )
    chain = find_best_linear_chain(graph, length=3, num_excitations=1)

    # chain[0] is excited first and sits idle the longest, so it must be q2 (the stable qubit)
    assert chain[0] == q2
    assert chain[-1] == q0


def test_grid_selection_prefers_distinct_readout_diagonals():
    """Verify that candidate paths with identical gate errors prefer distinct readout diagonals."""
    # Path 1: (0, 0) - (0, 1) - (0, 2) -> diagonals: 0, 1, 2 (all distinct)
    # Path 2: (1, 1) - (2, 0) - (2, 1) -> diagonals: 2, 2, 3 (collision on diagonal 2)
    p1_nodes = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)]
    p2_nodes = [cirq.GridQubit(1, 1), cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)]

    cz_errors = {
        (p1_nodes[0], p1_nodes[1]): (0.002, 0.0),
        (p1_nodes[1], p1_nodes[2]): (0.002, 0.0),
        (p2_nodes[0], p2_nodes[1]): (0.002, 0.0),
        (p2_nodes[1], p2_nodes[2]): (0.002, 0.0),
    }
    sq_errors = {q: (0.0004, 0.0) for q in (p1_nodes + p2_nodes)}

    graph = build_connectivity_graph(
        cz_errors,
        sq_pauli_error_rate=sq_errors,
        num_qubits=3,
        num_excitations=1,
    )
    chain = find_best_linear_chain(graph, length=3, num_excitations=1, readout_line_penalty_weight=0.02)
    # Path 1 should be preferred because it avoids collision on diagonal 2
    assert set(chain) == set(p1_nodes)
