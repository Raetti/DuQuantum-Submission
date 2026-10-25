"""Scoring and validation functions for Dicke state preparation.

Matches the official Google Quantum AI challenge scoring specifications.
"""

import itertools
import math
import numpy as np
import cirq


def create_dicke_statevector(num_qubits: int, num_excitations: int) -> np.ndarray:
    """Generate the exact statevector for an N-qubit, weight-n Dicke state."""
    psi = np.zeros(2**num_qubits, dtype=np.complex128)
    normalization = 1.0 / np.sqrt(math.comb(num_qubits, num_excitations))
    qubits = range(num_qubits)
    for excited_qubits in itertools.combinations(qubits, num_excitations):
        idx = sum(2**q for q in excited_qubits)
        psi[idx] = normalization
    return psi


def test_circuit(
    circuit: cirq.Circuit,
    num_excitations: int,
    qubits: set[cirq.GridQubit],
    sampler: cirq.Simulator = None,
) -> bool:
    """Verify that a circuit prepares the exact Dicke state on the specified qubits."""
    if sampler is None:
        sampler = cirq.Simulator()
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    num_qubits = len(qubits)
    keep_indices = list(range(num_qubits))

    psi_circuit_full = sampler.simulate(circuit, qubit_order=ordered_qubits).final_state_vector
    psi_circuit = (
        cirq.sub_state_vector(
            state_vector=psi_circuit_full,
            keep_indices=keep_indices,
        )
        if len(all_qubits) > len(qubits)
        else psi_circuit_full
    )

    ideal_psi = create_dicke_statevector(num_qubits, num_excitations)
    fidelity = float(np.abs(np.vdot(psi_circuit, ideal_psi)) ** 2)
    assert np.isclose(fidelity, 1.0, atol=1e-5), f"Fidelity is {fidelity}, expected 1.0"
    return True


def check_z_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int
) -> float:
    """Measure how close the observed Z-basis bitstring distribution is to ideal.

    The lower the score, the better.
    """
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(measured_excitations == num_excitations)

    if fraction_in_subspace == 0:
        return float("inf")

    chi2 = 0.0
    ideal_prob = 1.0 / math.comb(n_qubits, num_excitations)
    for indices in itertools.combinations(range(n_qubits), num_excitations):
        ideal_state = np.zeros(n_qubits)
        ideal_state[np.array(indices)] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = n_shots * fraction_in_subspace * ideal_prob
        chi2 += (match_count - expected_count) ** 2 / expected_count

    return float(ideal_prob * chi2 / fraction_in_subspace)


def get_correlator_average(bitstrings: np.ndarray, correlator_size: int) -> float:
    """Compute average of all correlators of weight correlator_size."""
    correlator_sum = 0.0
    n_shots, n_qubits = bitstrings.shape
    for indices in itertools.combinations(range(n_qubits), correlator_size):
        correlator_sum += 1.0 - 2.0 * np.mean(np.sum(bitstrings[:, indices], axis=1) % 2)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int, max_correlator_size: int
) -> float:
    """Check expected parity correlations in the X basis.

    The lower the score, the better.
    """
    comb = lambda n, k: math.comb(n, k) if (k >= 0 and n >= k) else 0
    n_shots, n_qubits = bitstrings.shape
    score = 0.0
    for correlator_size in range(1, max_correlator_size + 1):
        expected_correlator = (
            comb(correlator_size, correlator_size // 2)
            * comb(n_qubits - correlator_size, num_excitations - correlator_size // 2)
            / comb(n_qubits, num_excitations)
            if correlator_size % 2 == 0
            else 0.0
        )
        measured_correlator = get_correlator_average(bitstrings, correlator_size)
        score += (
            (expected_correlator - measured_correlator) ** 2 / expected_correlator**2
            if expected_correlator != 0
            else (expected_correlator - measured_correlator) ** 2
        )

    return float(score)


def overall_score(
    z_bitstrings: np.ndarray,
    x_bitstrings: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> float:
    """Overall benchmark score combining Z and X basis bitstrings.

    The higher the score, the better.
    """
    z_score = check_z_bitstring_distribution(z_bitstrings, num_excitations)
    x_score = check_x_bitstring_distribution(
        x_bitstrings, num_excitations, max_correlator_size
    )
    return 1.0 / (z_score + x_score)
