# -*- coding: utf-8 -*-
"""DuQuantum 2026 - Google Quantum AI Challenge Submission.

Self-contained judge submission file implementing Divide-and-Conquer Dicke state
preparation (Aktar, Bärtschi, Badawy, and Eidenbenz, arXiv:2112.12435) with
calibration-aware 2D Ladder and 1D LNN grid selection on Google Willow105.

Imports only cirq, cirq_google, their dependencies (numpy, networkx), and
Python's standard library.
"""

import itertools
import math
from typing import Optional, Sequence

import cirq
import cirq_google
import networkx as nx
import numpy as np


def _to_phased_xz(op: cirq.Operation, atol: float = 1e-12) -> Optional[cirq.Operation]:
    if isinstance(op.gate, cirq.ResetChannel):
        return op
    u = cirq.unitary(op, None)
    if u is None:
        raise ValueError(f"Non-unitary 1Q operation: {op}")
    gate = cirq.PhasedXZGate.from_matrix(u)
    if np.isclose(gate.x_exponent % 2.0, 0.0, atol=atol) and np.isclose(
        gate.z_exponent % 2.0, 0.0, atol=atol
    ):
        return None
    return gate.on(op.qubits[0])


def _lower_to_willow_gateset(circuit: cirq.Circuit, atol: float = 1e-12) -> cirq.Circuit:
    """Fuse adjacent 2Q blocks via KAK and merge 1Q gates into PhasedXZGate."""
    merged_2q = cirq.merge_k_qubit_unitaries(
        circuit,
        k=2,
        rewriter=lambda op: cirq.MatrixGate(cirq.unitary(op)).on(*op.qubits),
    )

    native_ops: list[cirq.Operation] = []
    for op in merged_2q.all_operations():
        if len(op.qubits) == 2:
            q0, q1 = op.qubits
            if (
                isinstance(op.gate, cirq.CZPowGate)
                and np.isclose(op.gate.exponent, 1.0, atol=atol)
                and np.isclose(op.gate.global_shift, 0.0, atol=atol)
            ):
                native_ops.append(cirq.CZ(q0, q1))
            else:
                u = cirq.unitary(op)
                kak_ops = cirq.two_qubit_matrix_to_cz_operations(
                    q0, q1, u, allow_partial_czs=False, atol=atol, clean_operations=True
                )
                for sub_op in kak_ops:
                    if len(sub_op.qubits) == 2:
                        native_ops.append(cirq.CZ(sub_op.qubits[0], sub_op.qubits[1]))
                    elif len(sub_op.qubits) == 1:
                        pxz = _to_phased_xz(sub_op, atol=atol)
                        if pxz is not None:
                            native_ops.append(pxz)
        elif len(op.qubits) == 1:
            pxz = _to_phased_xz(op, atol=atol)
            if pxz is not None:
                native_ops.append(pxz)

    inter = cirq.Circuit(native_ops)
    merged_1q = cirq.merge_single_qubit_gates_to_phxz(inter, atol=atol)

    clean_ops: list[cirq.Operation] = []
    for moment in merged_1q:
        for op in moment:
            if len(op.qubits) == 2:
                clean_ops.append(cirq.CZ(op.qubits[0], op.qubits[1]))
            elif len(op.qubits) == 1:
                pxz = _to_phased_xz(op, atol=atol)
                if pxz is not None:
                    clean_ops.append(pxz)

    return cirq.Circuit(clean_ops, strategy=cirq.InsertStrategy.EARLIEST)


def _u_be_conquer(wires: Sequence[cirq.Qid], k: int) -> cirq.Circuit:
    """Conquer unitary U^{be}_{m, k} (Figures 2 & 3 of Aktar et al., 2021)."""
    m = len(wires)
    c = cirq.Circuit()
    if m <= 1 or k <= 0:
        return c

    if k == 1:
        for i in range(m - 1):
            rem = m - i
            theta = 2.0 * math.acos(math.sqrt((rem - 1.0) / rem))
            c.append(cirq.ry(theta / 2.0)(wires[i + 1]))
            c.append(cirq.CNOT(wires[i], wires[i + 1]))
            c.append(cirq.ry(-theta / 2.0)(wires[i + 1]))
            c.append(cirq.CNOT(wires[i + 1], wires[i]))
        return c

    k_eff = min(k, m - 1)
    t1 = 2.0 * math.acos(math.sqrt((m - 1.0) / m))
    c.append(
        [
            cirq.ry(math.pi / 2.0)(wires[1]),
            cirq.CNOT(wires[1], wires[0]),
            cirq.ry(t1 / 2.0)(wires[0]),
            cirq.ry(t1 / 2.0)(wires[1]),
            cirq.CNOT(wires[1], wires[0]),
            cirq.ry(-math.pi / 2.0)(wires[1]),
        ]
    )
    for l_idx in range(2, k_eff + 1):
        tl = 2.0 * math.acos(math.sqrt(float(l_idx) / m))
        w_a, w_b, w_c = wires[l_idx - 2], wires[l_idx - 1], wires[l_idx]
        c.append(
            [
                cirq.CNOT(w_b, w_c),
                cirq.ry(-tl / 4.0)(w_b),
                cirq.CNOT(w_a, w_b),
                cirq.ry(tl / 4.0)(w_b),
                cirq.CNOT(w_c, w_b),
                cirq.ry(-tl / 4.0)(w_b),
                cirq.CNOT(w_a, w_b),
                cirq.ry(tl / 4.0)(w_b),
                cirq.CNOT(w_b, w_c),
            ]
        )

    if k_eff < m - 1:
        for pos in range(k_eff, m - 1):
            c.append(cirq.CNOT(wires[pos], wires[pos + 1]))
            c.append(cirq.CNOT(wires[pos + 1], wires[pos]))

    next_k = min(k, m - 1)
    if k >= 2 and m - 1 == 2:
        next_k = 2
    c.append(_u_be_conquer(wires[: m - 1], next_k))
    return c


def _u_padded_top_conquer(wires: Sequence[cirq.Qid], k: int) -> cirq.Circuit:
    """Conquer unitary U_{m, k} when wires[:m-k] are |0> and wires[m-k:] hold (q_1, ..., q_k)."""
    m = len(wires)
    if m <= k or k <= 0:
        return _u_be_conquer(wires, k)
    if k == 1:
        return _u_be_conquer(list(reversed(wires)), 1)

    c = cirq.Circuit()
    for idx in range(m - k - 1, -1, -1):
        for pos in range(idx, idx + k):
            c.append(cirq.CNOT(wires[pos + 1], wires[pos]))
            c.append(cirq.CNOT(wires[pos], wires[pos + 1]))
    c.append(_u_be_conquer(wires, k))
    return c


def _build_ladder_dicke(
    upper: Sequence[cirq.Qid],
    lower: Sequence[cirq.Qid],
    num_excitations: int,
) -> cirq.Circuit:
    """Ladder Divide-and-Conquer Dicke circuit (Section III-B, Figure 4 of Aktar et al., 2021)."""
    n1 = len(upper)
    n2 = len(lower)
    n_total = n1 + n2
    all_q = list(upper) + list(lower)

    if num_excitations == 0:
        return cirq.Circuit()
    if num_excitations == n_total:
        return _lower_to_willow_gateset(cirq.Circuit(cirq.X(q) for q in all_q))

    invert = num_excitations > n_total // 2
    k = (n_total - num_excitations) if invert else num_excitations

    c = cirq.Circuit()
    x_weights = [
        math.comb(n1, i) * math.comb(n2, k - i) if (i <= n1 and (k - i) <= n2) else 0
        for i in range(k + 1)
    ]
    suffix = [sum(x_weights[i:]) for i in range(k + 1)]

    if suffix[0] > 0:
        theta0 = 2.0 * math.acos(math.sqrt(x_weights[0] / suffix[0]))
        c.append(cirq.ry(theta0)(upper[0]))

    for i in range(1, k):
        if suffix[i] > 0:
            phi = 2.0 * math.acos(math.sqrt(suffix[i + 1] / suffix[i]))
            c.append(cirq.ry(phi / 2.0)(upper[i]))
            c.append(cirq.CNOT(upper[i - 1], upper[i]))
            c.append(cirq.ry(-phi / 2.0)(upper[i]))

    for j in range(k):
        c.append(cirq.X(lower[k - 1 - j]))
        c.append(cirq.CNOT(upper[j], lower[k - 1 - j]))

    c.append(_u_be_conquer(upper, k))
    c.append(_u_be_conquer(lower, k))

    if invert:
        for q in all_q:
            c.append(cirq.X(q))

    return _lower_to_willow_gateset(c)


def _build_lnn_dicke(
    qubits: Sequence[cirq.Qid],
    num_excitations: int,
) -> cirq.Circuit:
    """LNN Divide-and-Conquer Dicke circuit (Section III-C, Figures 5-6 of Aktar et al., 2021)."""
    n_total = len(qubits)
    if num_excitations == 0:
        return cirq.Circuit()
    if num_excitations == n_total:
        return _lower_to_willow_gateset(cirq.Circuit(cirq.X(q) for q in qubits))

    invert = num_excitations > n_total // 2
    k = (n_total - num_excitations) if invert else num_excitations

    c = cirq.Circuit()
    n1 = n_total // 2
    n2 = n_total - n1
    upper_block = qubits[:n1]
    lower_block = qubits[n1:]

    x_weights = [
        math.comb(n1, i) * math.comb(n2, k - i) if (i <= n1 and (k - i) <= n2) else 0
        for i in range(k + 1)
    ]
    suffix = [sum(x_weights[i:]) for i in range(k + 1)]

    u_active = qubits[n1 - k : n1]
    l_active = qubits[n1 : n1 + k]

    theta0 = 2.0 * math.acos(math.sqrt(x_weights[0] / suffix[0]))
    c.append(cirq.ry(theta0)(u_active[0]))
    for i in range(1, k):
        phi = 2.0 * math.acos(math.sqrt(suffix[i + 1] / suffix[i]))
        c.append(cirq.ry(phi / 2.0)(u_active[i]))
        c.append(cirq.CNOT(u_active[i - 1], u_active[i]))
        c.append(cirq.ry(-phi / 2.0)(u_active[i]))

    for j in range(k):
        c.append(cirq.X(l_active[j]))

    if k == 2 and n1 == 2 and n2 == 2:
        c.append(cirq.CNOT(u_active[1], l_active[0]))
        c.append(cirq.CNOT(l_active[0], l_active[1]))
        c.append(cirq.SWAP(u_active[1], l_active[0]))
        c.append(cirq.CNOT(l_active[0], l_active[1]))
        c.append(cirq.SWAP(u_active[1], l_active[0]))
    else:
        for s in range(k):
            idx_a = n1 - 1 - s
            for pos in range(idx_a, idx_a + 2 * s + 1):
                c.append(cirq.CNOT(qubits[pos], qubits[pos + 1]))
            for pos in range(idx_a + 2 * s - 1, idx_a - 1, -1):
                c.append(cirq.CNOT(qubits[pos], qubits[pos + 1]))
                c.append(cirq.CNOT(qubits[pos + 1], qubits[pos]))

    c.append(_u_padded_top_conquer(upper_block, k))
    c.append(_u_be_conquer(lower_block, k))

    if invert:
        for q in qubits:
            c.append(cirq.X(q))

    return _lower_to_willow_gateset(c)


def _build_connectivity_graph(
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
) -> nx.Graph:
    g = nx.Graph()
    if sq_pauli_error_rate:
        for q, (err, _) in sq_pauli_error_rate.items():
            g.add_node(q, error=float(err))
    for (q1, q2), (cz_err, _) in cz_pauli_error_rate.items():
        if q1 not in g:
            g.add_node(q1, error=0.001)
        if q2 not in g:
            g.add_node(q2, error=0.001)
        sq1 = g.nodes[q1].get("error", 0.001)
        sq2 = g.nodes[q2].get("error", 0.001)
        g.add_edge(q1, q2, weight=float(cz_err) + 0.5 * (sq1 + sq2), cz_error=float(cz_err))
    return g


def _extract_template_workload(
    template: cirq.Circuit,
    abs_qubits: Sequence[cirq.Qid],
) -> tuple[list[tuple[int, int, int]], list[int], list[int], int]:
    q_to_idx = {q: i for i, q in enumerate(abs_qubits)}
    n_q = len(abs_qubits)
    edge_counts: dict[tuple[int, int], int] = {}
    sq_counts = [0] * n_q
    active_moments = [0] * n_q
    n_moments = len(template)

    for moment in template:
        for op in moment:
            if len(op.qubits) == 2:
                i0, i1 = q_to_idx[op.qubits[0]], q_to_idx[op.qubits[1]]
                key = (i0, i1) if i0 < i1 else (i1, i0)
                edge_counts[key] = edge_counts.get(key, 0) + 1
                active_moments[i0] += 1
                active_moments[i1] += 1
            elif len(op.qubits) == 1:
                idx = q_to_idx[op.qubits[0]]
                sq_counts[idx] += 1
                active_moments[idx] += 1

    idle_moments = [max(0, n_moments - act) for act in active_moments]
    edge_list = [(i0, i1, cnt) for (i0, i1), cnt in edge_counts.items()]
    return edge_list, sq_counts, idle_moments, n_moments


def _build_calibration_costs(
    graph: nx.Graph,
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[
    dict[tuple[cirq.GridQubit, cirq.GridQubit], float],
    dict[cirq.GridQubit, float],
    dict[cirq.GridQubit, float],
    dict[cirq.GridQubit, float],
    dict[cirq.GridQubit, float],
]:
    p1 = max(0.0, min(1.0, num_excitations / num_qubits)) if num_qubits > 0 else 0.25
    p0 = 1.0 - p1

    edge_cost: dict[tuple[cirq.GridQubit, cirq.GridQubit], float] = {}
    for (q0, q1), (cz_err, _) in cz_pauli_error_rate.items():
        w = -math.log(max(1.0 - float(cz_err), 1e-6))
        edge_cost[(q0, q1)] = w
        edge_cost[(q1, q0)] = w

    sq_cost: dict[cirq.GridQubit, float] = {}
    ro_cost: dict[cirq.GridQubit, float] = {}
    idle_rate: dict[cirq.GridQubit, float] = {}
    spectator_cost: dict[cirq.GridQubit, float] = {}

    for q in graph.nodes():
        sq_err = float(sq_pauli_error_rate[q][0]) if q in sq_pauli_error_rate else 4e-4
        e01 = float(zero_errors[q][0]) if q in zero_errors else 5e-3
        e10 = float(one_errors[q][0]) if q in one_errors else 1e-2

        ro_err = p0 * e01 + p1 * e10
        sc = -math.log(max(1.0 - sq_err, 1e-6))
        rc = -math.log(max(1.0 - ro_err, 1e-6))
        t1_proxy = -math.log(max(1.0 - e10, 1e-6))

        sq_cost[q] = sc
        ro_cost[q] = rc
        idle_rate[q] = 0.08 * sc + 0.015 * t1_proxy

        inc = sum(edge_cost.get((q, nbr), 0.002) for nbr in graph.neighbors(q))
        spectator_cost[q] = 0.015 * inc

    return edge_cost, sq_cost, ro_cost, idle_rate, spectator_cost


def _score_physical_embedding(
    phys_qubits: Sequence[cirq.GridQubit],
    edge_list: list[tuple[int, int, int]],
    sq_counts: list[int],
    idle_moments: list[int],
    n_moments: int,
    edge_cost: dict[tuple[cirq.GridQubit, cirq.GridQubit], float],
    sq_cost: dict[cirq.GridQubit, float],
    ro_cost: dict[cirq.GridQubit, float],
    idle_rate: dict[cirq.GridQubit, float],
    spectator_cost: dict[cirq.GridQubit, float],
) -> float:
    cost = 0.0
    for i0, i1, mult in edge_list:
        cost += mult * edge_cost.get((phys_qubits[i0], phys_qubits[i1]), 1.0)

    for idx, q in enumerate(phys_qubits):
        cost += (
            sq_counts[idx] * sq_cost.get(q, 4e-4)
            + 0.08 * ro_cost.get(q, 1e-2)
            + idle_moments[idx] * idle_rate.get(q, 1e-4)
            + spectator_cost.get(q, 0.0)
        )

    cost += 0.00015 * n_moments * max(len(phys_qubits), 1)
    return cost


def _find_best_ladder_embedding(
    graph: nx.Graph,
    n1: int,
    n2: int,
    k: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
    num_excitations: int,
) -> Optional[tuple[cirq.Circuit, set[cirq.GridQubit], float]]:
    abs_upper = [cirq.LineQubit(i) for i in range(n1)]
    abs_lower = [cirq.LineQubit(n1 + i) for i in range(n2)]
    abs_all = abs_upper + abs_lower
    template = _build_ladder_dicke(abs_upper, abs_lower, num_excitations)
    edge_list, sq_counts, idle_moments, n_moments = _extract_template_workload(template, abs_all)
    edge_cost, sq_cost, ro_cost, idle_rate, spectator_cost = _build_calibration_costs(
        graph,
        n1 + n2,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )

    nodes = sorted(graph.nodes())
    node_set = set(nodes)

    cores: list[tuple[list[cirq.GridQubit], list[cirq.GridQubit]]] = []
    for q in nodes:
        r, c = q.row, q.col
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for perp_r, perp_c in [(-dc, dr), (dc, -dr)]:
                u_core = [cirq.GridQubit(r + i * dr, c + i * dc) for i in range(k)]
                l_core = [
                    cirq.GridQubit(
                        r + perp_r + (k - 1 - i) * dr,
                        c + perp_c + (k - 1 - i) * dc,
                    )
                    for i in range(k)
                ]
                if not (all(u in node_set for u in u_core) and all(l in node_set for l in l_core)):
                    continue
                if len(set(u_core) | set(l_core)) != 2 * k:
                    continue
                valid = True
                for i in range(k - 1):
                    if not (
                        graph.has_edge(u_core[i], u_core[i + 1])
                        and graph.has_edge(l_core[i], l_core[i + 1])
                    ):
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(k):
                    if not graph.has_edge(u_core[j], l_core[k - 1 - j]):
                        valid = False
                        break
                if valid:
                    cores.append((u_core, l_core))

    def extend_paths(
        start: cirq.GridQubit, length: int, forbidden: set[cirq.GridQubit]
    ) -> list[list[cirq.GridQubit]]:
        if length == 0:
            return [[]]
        results: list[list[cirq.GridQubit]] = []

        def dfs(curr: cirq.GridQubit, path: list[cirq.GridQubit], used: set[cirq.GridQubit]):
            if len(results) >= 12:
                return
            if len(path) == length:
                results.append(list(path))
                return
            nbrs = sorted(
                (n for n in graph.neighbors(curr) if n not in used),
                key=lambda n: edge_cost.get((curr, n), 0.01) + 0.08 * ro_cost.get(n, 0.01),
            )
            for nb in nbrs[:4]:
                used.add(nb)
                path.append(nb)
                dfs(nb, path, used)
                path.pop()
                used.remove(nb)
                if len(results) >= 12:
                    return

        dfs(start, [], set(forbidden))
        return results

    tail_u_len = n1 - k
    tail_l_len = n2 - k

    best_score = float("inf")
    best_phys: Optional[list[cirq.GridQubit]] = None

    for u_core, l_core in cores:
        base_used = set(u_core) | set(l_core)
        u_tails = extend_paths(u_core[-1], tail_u_len, base_used)
        if not u_tails:
            continue
        for u_tail in u_tails:
            used_with_u = base_used | set(u_tail)
            l_tails = extend_paths(l_core[-1], tail_l_len, used_with_u)
            for l_tail in l_tails:
                phys_q = u_core + u_tail + l_core + l_tail
                sc = _score_physical_embedding(
                    phys_q,
                    edge_list,
                    sq_counts,
                    idle_moments,
                    n_moments,
                    edge_cost,
                    sq_cost,
                    ro_cost,
                    idle_rate,
                    spectator_cost,
                )
                if sc < best_score:
                    best_score = sc
                    best_phys = phys_q

    if best_phys is None:
        return None

    q_map = dict(zip(abs_all, best_phys))
    mapped = template.transform_qubits(lambda q: q_map[q])
    return mapped, set(best_phys), best_score


def _find_best_lnn_embedding(
    graph: nx.Graph,
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit], float]:
    abs_q = [cirq.LineQubit(i) for i in range(num_qubits)]
    template = _build_lnn_dicke(abs_q, num_excitations)
    edge_list, sq_counts, idle_moments, n_moments = _extract_template_workload(template, abs_q)
    edge_cost, sq_cost, ro_cost, idle_rate, spectator_cost = _build_calibration_costs(
        graph,
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )

    best_score = float("inf")
    best_chain: Optional[list[cirq.GridQubit]] = None
    eval_count = 0
    max_evals = 150

    nodes = sorted(
        graph.nodes(),
        key=lambda q: sq_cost.get(q, 4e-4)
        + 0.08 * ro_cost.get(q, 1e-2)
        + idle_moments[0] * idle_rate.get(q, 1e-4)
        + sum(edge_cost.get((q, nbr), 0.002) for nbr in graph.neighbors(q))
        / max(len(list(graph.neighbors(q))), 1),
    )

    def dfs(curr: cirq.GridQubit, path: list[cirq.GridQubit], used: set[cirq.GridQubit]):
        nonlocal best_score, best_chain, eval_count
        eval_count += 1
        if eval_count > max_evals:
            return
        if len(path) == num_qubits:
            for candidate in (path, list(reversed(path))):
                sc = _score_physical_embedding(
                    candidate,
                    edge_list,
                    sq_counts,
                    idle_moments,
                    n_moments,
                    edge_cost,
                    sq_cost,
                    ro_cost,
                    idle_rate,
                    spectator_cost,
                )
                if sc < best_score:
                    best_score = sc
                    best_chain = list(candidate)
            return

        nbrs = sorted(
            (n for n in graph.neighbors(curr) if n not in used),
            key=lambda n: edge_cost.get((curr, n), 0.01)
            + 0.08 * ro_cost.get(n, 0.01)
            + sq_cost.get(n, 4e-4),
        )
        for nb in nbrs[:4]:
            used.add(nb)
            path.append(nb)
            dfs(nb, path, used)
            path.pop()
            used.remove(nb)
            if eval_count > max_evals:
                return

    for start in nodes[:40]:
        eval_count = 0
        dfs(start, [start], {start})

    if best_chain is None:
        raise NotImplementedError(f"Could not embed linear chain of length {num_qubits}.")

    q_map = dict(zip(abs_q, best_chain))
    mapped = template.transform_qubits(lambda q: q_map[q])
    return mapped, set(best_chain), best_score


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[
        tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]
    ],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state.

    Args:
        num_qubits: The number, N, of qubits in the Dicke state.
        num_excitations: The number, n, of 1s in the state.
        cz_pauli_error_rate: Measured CZ Pauli error rates and their statistical uncertainties.
        sq_pauli_error_rate: Measured single-qubit Pauli error rates and their statistical uncertainties.
        zero_errors: Measured parallel 0 -> 1 readout error rates and their statistical uncertainties.
        one_errors: Measured 1 -> 0 readout error rates and their statistical uncertainties.

    Returns:
        A Cirq circuit to produce Dicke state and a set indicating which qubits the resulting state lives on.

    Raises:
        NotImplementedError: The requested (N, n) combination is not implemented.
    """
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"Invalid excitations {num_excitations} for {num_qubits} qubits."
        )

    if num_qubits == 0:
        return cirq.Circuit(), set()

    graph = _build_connectivity_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    k = min(num_excitations, num_qubits - num_excitations)

    if k == 0:
        ranked = sorted(
            graph.nodes(),
            key=lambda q: float(sq_pauli_error_rate.get(q, (0.001, 0.0))[0])
            + 0.02
            * (
                float(zero_errors.get(q, (0.01, 0.0))[0])
                + float(one_errors.get(q, (0.01, 0.0))[0])
            ),
        )
        if len(ranked) < num_qubits:
            raise NotImplementedError(f"Insufficient working qubits for N={num_qubits}.")
        chosen = ranked[:num_qubits]
        if num_excitations == 0:
            return cirq.Circuit(), set(chosen)
        c = _lower_to_willow_gateset(cirq.Circuit(cirq.X(q) for q in chosen))
        return c, set(chosen)

    best_circuit: Optional[cirq.Circuit] = None
    best_qubits: Optional[set[cirq.GridQubit]] = None
    best_score = float("inf")

    if k >= 2 and num_qubits >= 4:
        n1 = num_qubits // 2
        n2 = num_qubits - n1
        ladder_res = _find_best_ladder_embedding(
            graph,
            n1,
            n2,
            k,
            cz_pauli_error_rate,
            sq_pauli_error_rate,
            zero_errors,
            one_errors,
            num_excitations,
        )
        if ladder_res is not None:
            best_circuit, best_qubits, best_score = ladder_res

    if best_circuit is None or k == 1:
        lnn_circuit, lnn_qubits, lnn_score = _find_best_lnn_embedding(
            graph,
            num_qubits,
            num_excitations,
            cz_pauli_error_rate,
            sq_pauli_error_rate,
            zero_errors,
            one_errors,
        )
        if lnn_score < best_score:
            best_circuit, best_qubits, best_score = lnn_circuit, lnn_qubits, lnn_score

    if best_circuit is None or best_qubits is None or len(best_circuit) > 1750:
        raise NotImplementedError(
            f"Could not construct valid circuit within 1750 moments for N={num_qubits}, n={num_excitations}."
        )
    return best_circuit, best_qubits


def create_dicke_statevector(num_qubits: int, num_excitations: int) -> np.ndarray:
    psi = np.zeros(2**num_qubits)
    normalization = 1 / np.sqrt(math.comb(num_qubits, num_excitations))
    qubits = range(num_qubits)
    for excited_qubits in itertools.combinations(qubits, num_excitations):
        idx = sum(2**qubit_number for qubit_number in excited_qubits)
        psi[idx] = normalization
    return psi


def test_circuit(
    circuit: cirq.Circuit,
    num_excitations: int,
    qubits: set[cirq.GridQubit],
    sampler=cirq.Simulator(),
):
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    keep_indices = range(len(qubits))
    num_qubits = len(qubits)
    psi_circuit_full = sampler.simulate(circuit, qubit_order=ordered_qubits).final_state_vector
    psi_circuit = (
        cirq.sub_state_vector(state_vector=psi_circuit_full, keep_indices=keep_indices)
        if len(all_qubits) > len(qubits)
        else psi_circuit_full
    )
    assert np.isclose(
        np.abs(psi_circuit @ create_dicke_statevector(num_qubits, num_excitations)) ** 2,
        1.0,
    )


def check_z_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int
) -> float:
    """Measure how close the observed Z-basis distribution of bitstrings is to
    the ideal one. The lower the score, the better.
    """
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(measured_excitations == num_excitations)

    chi2 = 0
    ideal_prob = 1 / math.comb(n_qubits, num_excitations)
    for indices in itertools.combinations(range(n_qubits), num_excitations):
        ideal_state = np.zeros(n_qubits)
        ideal_state[np.array(indices)] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = n_shots * fraction_in_subspace * ideal_prob
        chi2 += (match_count - expected_count) ** 2 / expected_count

    return ideal_prob * chi2 / fraction_in_subspace


def get_correlator_average(bitstrings: np.ndarray, correlator_size: int) -> float:
    """Get the average of all weight-correlator_size correlators in `bitstrings`."""
    correlator_sum = 0.0
    n_shots, n_qubits = bitstrings.shape
    for indices in itertools.combinations(range(n_qubits), correlator_size):
        correlator_sum += 1 - 2 * np.mean(np.sum(bitstrings[:, indices], axis=1) % 2)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int, max_correlator_size: int
) -> float:
    """Check expected correlations in the X basis. The lower the score, the better."""
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

    return score


def overall_score(
    z_bitstrings: np.ndarray,
    x_bitstrings: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> float:
    """An overall score that checks how the Z and X basis bitstrings compare to their ideal
    distributions. The higher the score, the better.
    """
    return 1 / (
        check_z_bitstring_distribution(z_bitstrings, num_excitations)
        + check_x_bitstring_distribution(
            x_bitstrings, num_excitations, max_correlator_size
        )
    )


def pick_single_qubit_dropouts(
    dropout_rate: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> set[cirq.GridQubit]:
    """Pick some qubits to be broken."""
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(
    median: float,
    spread: float,
    dropouts: set[cirq.GridQubit],
    uncertainty: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> dict[cirq.GridQubit, tuple[float, float]]:
    """Generate fake single-qubit error rates."""
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_working = np.array([q not in dropouts for q in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rate = rng.lognormal(np.log(median), spread, len(working_qubits))
    return dict(
        zip(working_qubits, zip(error_rate, uncertainty * np.ones(len(working_qubits))))
    )


def generate_two_qubit_pauli_error_rates(
    median: float = 2.5e-3,
    spread: float = 0.5,
    dropout_rate: float = 0.01,
    uncertainty: float = 5e-4,
    rng: np.random.Generator = np.random.default_rng(),
    single_qubit_dropouts: set[cirq.GridQubit] = None,
) -> dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]]:
    """Generate fake two-qubit error rates."""
    if single_qubit_dropouts is None:
        single_qubit_dropouts = set()

    all_pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    no_qubit_dropouts = np.array(
        [np.all([q not in single_qubit_dropouts for q in pair]) for pair in all_pairs]
    )
    is_working = (rng.random(len(all_pairs)) > dropout_rate) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rate = rng.lognormal(np.log(median), spread, len(working_pairs))
    return dict(
        zip(working_pairs, zip(error_rate, uncertainty * np.ones(len(working_pairs))))
    )


def create_dicke_state_adaptive(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state, allowing for adaptive circuits."""
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"Invalid excitations {num_excitations} for {num_qubits} qubits."
        )

    device = cirq_google.Willow105
    all_pairs = list(device.metadata.qubit_pairs)
    all_qubits = list(device.metadata.qubit_set)
    cz_dummy = {tuple(pair): (0.002, 0.0005) for pair in all_pairs}
    sq_dummy = {q: (0.0005, 0.0001) for q in all_qubits}
    ro_dummy = {q: (0.005, 0.001) for q in all_qubits}

    return create_dicke_state(
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        cz_pauli_error_rate=cz_dummy,
        sq_pauli_error_rate=sq_dummy,
        zero_errors=ro_dummy,
        one_errors=ro_dummy,
    )


def validate_circuit(
    circuit: cirq.Circuit,
    num_excitations: int,
    qubits: set[cirq.GridQubit],
) -> None:
    cirq_google.Willow105.validate_circuit(circuit)
    if len(qubits) <= 10:
        test_circuit(circuit, num_excitations, qubits)


def score_solution() -> float:
    score = 0.0
    for num_qubits in range(2, 21):
        for num_excitations in range(1, num_qubits // 2 + 1):
            try:
                circuit, qubits = create_dicke_state_adaptive(num_qubits, num_excitations)
            except NotImplementedError:
                continue
            validate_circuit(circuit, num_excitations, qubits)
            score += num_qubits * num_excitations / len(circuit)
    return score


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    single_qubit_dropouts = pick_single_qubit_dropouts(0.01, rng=rng)
    sq_pauli_error_rate = generate_qubit_error_rates(
        4e-4, 0.5, single_qubit_dropouts, 2e-4, rng=rng
    )
    cz_pauli_error_rate = generate_two_qubit_pauli_error_rates(
        single_qubit_dropouts=single_qubit_dropouts, rng=rng
    )
    zero_errors = generate_qubit_error_rates(
        5e-3, 0.5, single_qubit_dropouts, 2e-4, rng=rng
    )
    one_errors = generate_qubit_error_rates(
        1e-2, 0.5, single_qubit_dropouts, 3e-4, rng=rng
    )

    for N in range(2, 9):
        for n in range(1, N // 2 + 1):
            c, qs = create_dicke_state(
                N,
                n,
                cz_pauli_error_rate,
                sq_pauli_error_rate,
                zero_errors,
                one_errors,
            )
            cirq_google.Willow105.validate_circuit(c)
            test_circuit(c, n, qs)
            cz_count = sum(1 for op in c.all_operations() if len(op.qubits) == 2)
            print(
                f"Verified |D^({N})_{n}>: {cz_count:2d} CZs, {len(c):2d} moments on {len(qs)} Willow qubits (fidelity = 1.0)"
            )
