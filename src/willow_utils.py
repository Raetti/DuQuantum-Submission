"""Willow topology utilities, graph embedding, and readout line constraint checks."""

import math
from typing import Iterable, Optional
import networkx as nx
import cirq


def _compute_circuit_workload(
    num_qubits: int,
    num_excitations: int,
) -> tuple[list[float], list[float], list[float]]:
    """Compute per-position workloads for an N-qubit, weight-n Dicke state circuit.

    Returns:
        coupler_cz_counts: Number of CZ gates expected on each adjacent link (length N-1).
        sq_gate_counts: Number of active single-qubit gates on each qubit (length N).
        idle_weights: Idle duration weighting factor on each qubit (length N).
    """
    if num_qubits <= 1:
        return [], [1.0] * num_qubits, [0.0] * num_qubits

    n_links = num_qubits - 1

    # Default Givens-rotation cascade (e.g. W-state n=1 or bit-flipped n=N-1)
    if num_excitations in (1, num_qubits - 1) or num_qubits <= 3:
        # Each link executes 1 Givens rotation (4 CZ gates in current synthesis)
        coupler_cz = [4.0] * n_links
        # Endpoints execute 1 Givens block + 1 basis gate; interior qubits execute 2 blocks + 1 basis gate
        sq_gates = [4.0] + [7.0] * (num_qubits - 2) + [4.0] if num_qubits > 2 else [4.0, 4.0]
        # In a left-to-right cascade: q0 is excited first and idles while cascade finishes;
        # q_{N-1} finishes last right before measurement.
        idle_weights = [float(max(0, num_qubits - 1 - max(1, i))) for i in range(num_qubits)]
        return coupler_cz, sq_gates, idle_weights

    # Specialized case for N=4, n=2 (recursive controlled-swap decomposition)
    if num_qubits == 4 and num_excitations == 2:
        coupler_cz = [7.0, 11.0, 5.0]
        sq_gates = [6.0, 10.0, 9.0, 5.0]
        idle_weights = [3.0, 2.0, 1.0, 0.0]
        return coupler_cz, sq_gates, idle_weights

    # General Dicke cascade fallback
    coupler_cz = [4.0] * n_links
    sq_gates = [5.0] * num_qubits
    idle_weights = [float(num_qubits - 1 - i) for i in range(num_qubits)]
    return coupler_cz, sq_gates, idle_weights


def build_connectivity_graph(
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
    zero_errors: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
    one_errors: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
    num_qubits: Optional[int] = None,
    num_excitations: Optional[int] = None,
) -> nx.Graph:
    """Construct an undirected graph of available qubits and CZ couplers.

    Incorporates CZ infidelity, single-qubit error rates, and state-dependent
    readout error rates (0->1 and 1->0) for the Dicke state |D^{(N)}_n>.
    """
    g = nx.Graph()

    # Determine marginal excitation probabilities for Dicke state |D^{(N)}_n>
    if num_qubits and num_qubits > 0 and num_excitations is not None:
        p1 = max(0.0, min(1.0, num_excitations / num_qubits))
    else:
        p1 = 0.25
    p0 = 1.0 - p1

    # Collect all available qubits
    all_qubits: set[cirq.GridQubit] = set()
    if sq_pauli_error_rate:
        all_qubits.update(sq_pauli_error_rate.keys())
    for q1, q2 in cz_pauli_error_rate.keys():
        all_qubits.add(q1)
        all_qubits.add(q2)

    for q in all_qubits:
        sq_err = sq_pauli_error_rate[q][0] if (sq_pauli_error_rate and q in sq_pauli_error_rate) else 4e-4
        e01 = zero_errors[q][0] if (zero_errors and q in zero_errors) else 5e-3
        e10 = one_errors[q][0] if (one_errors and q in one_errors) else 1e-2

        # Expected readout infidelity given Dicke state marginal statistics
        ro_err = p0 * e01 + p1 * e10

        ro_cost = -math.log(max(1.0 - ro_err, 1e-6))
        sq_cost = -math.log(max(1.0 - sq_err, 1e-6))
        # Idle cost rate: dephasing (correlated with sq_err) plus T1 relaxation into |0>
        idle_cost_rate = sq_cost + 0.2 * (-math.log(max(1.0 - e10, 1e-6)))

        g.add_node(
            q,
            error=sq_err,
            sq_cost=sq_cost,
            ro_err=ro_err,
            ro_cost=ro_cost,
            idle_cost_rate=idle_cost_rate,
            zero_err=e01,
            one_err=e10,
        )

    for (q1, q2), (cz_err, _) in cz_pauli_error_rate.items():
        weight = -math.log(max(1.0 - cz_err, 1e-6))
        g.add_edge(q1, q2, weight=weight, cz_err=cz_err)

    # Calculate small spectator crosstalk penalty for qubits with many active incident couplers
    for q in g.nodes():
        incident_weights = sum(g[q][nbr].get("weight", 0.002) for nbr in g[q])
        g.nodes[q]["spectator_cost"] = 0.05 * incident_weights

    return g


def _evaluate_directed_chain(
    path: list[cirq.GridQubit],
    graph: nx.Graph,
    coupler_cz_counts: list[float],
    sq_gate_counts: list[float],
    idle_weights: list[float],
    readout_line_penalty_weight: float = 0.015,
) -> float:
    """Calculate the exact total expected infidelity cost of an ordered path."""
    n = len(path)
    total_cost = 0.0

    # 1. Coupler CZ gate costs
    for i in range(n - 1):
        u, v = path[i], path[i + 1]
        edge_weight = graph[u][v].get("weight", 0.0025) if graph.has_edge(u, v) else 1.0
        cz_mult = coupler_cz_counts[i] if i < len(coupler_cz_counts) else 4.0
        total_cost += cz_mult * edge_weight

    # 2. Qubit node costs (readout, active single-qubit gates, idle stability, spectator)
    for i in range(n):
        q = path[i]
        node_data = graph.nodes[q]
        ro_cost = node_data.get("ro_cost", 0.01)
        sq_cost = node_data.get("sq_cost", 0.0004)
        idle_rate = node_data.get("idle_cost_rate", 0.001)
        spectator_cost = node_data.get("spectator_cost", 0.0)

        sq_mult = sq_gate_counts[i] if i < len(sq_gate_counts) else 4.0
        idle_mult = idle_weights[i] if i < len(idle_weights) else float(n - 1 - i)

        total_cost += ro_cost + (sq_mult * sq_cost) + (idle_mult * idle_rate) + spectator_cost

    # 3. Readout line multiplexing penalty (shared diagonal r + c)
    diag_counts: dict[int, int] = {}
    for q in path:
        diag = q.row + q.col
        diag_counts[diag] = diag_counts.get(diag, 0) + 1

    line_collisions = sum(count - 1 for count in diag_counts.values() if count > 1)
    total_cost += line_collisions * readout_line_penalty_weight

    return total_cost


def find_best_linear_chain(
    graph: nx.Graph,
    length: int,
    num_excitations: int = 1,
    readout_line_penalty_weight: float = 0.015,
    max_evaluations: int = 5000,
) -> list[cirq.GridQubit]:
    """Find a low-error simple path of specified length optimized for Dicke state preparation.

    Considers CZ gate workloads, state-dependent readout errors, position-dependent idle decay,
    and diagonal readout line collisions. Automatically chooses the optimal orientation
    (e.g. placing the more stable endpoint at q0).
    """
    if length > len(graph):
        raise ValueError(f"Requested path length {length} exceeds graph size {len(graph)}")

    if length == 1:
        return [min(graph.nodes(), key=lambda q: graph.nodes[q].get("ro_cost", 0.0) + graph.nodes[q].get("sq_cost", 0.0))]

    coupler_cz_counts, sq_gate_counts, idle_weights = _compute_circuit_workload(length, num_excitations)

    best_path: Optional[list[cirq.GridQubit]] = None
    best_cost = float("inf")
    eval_count = 0

    # Sort candidate starting nodes by initial suitability for q0
    candidate_starts = sorted(
        graph.nodes(),
        key=lambda n: (
            graph.nodes[n].get("ro_cost", 0.0)
            + graph.nodes[n].get("idle_cost_rate", 0.0) * (length - 1)
            + sum(graph[n][nbr].get("weight", 0.002) for nbr in graph[n]) / max(len(graph[n]), 1)
        ),
    )

    def consider_full_path(path: list[cirq.GridQubit]):
        nonlocal best_path, best_cost
        # Evaluate both forward and reverse orientations
        cost_fwd = _evaluate_directed_chain(
            path, graph, coupler_cz_counts, sq_gate_counts, idle_weights, readout_line_penalty_weight
        )
        path_rev = list(reversed(path))
        cost_rev = _evaluate_directed_chain(
            path_rev, graph, coupler_cz_counts, sq_gate_counts, idle_weights, readout_line_penalty_weight
        )

        if cost_fwd < best_cost:
            best_cost = cost_fwd
            best_path = list(path)
        if cost_rev < best_cost:
            best_cost = cost_rev
            best_path = list(path_rev)

    def dfs(current_path: list[cirq.GridQubit], current_edge_cost: float):
        nonlocal best_path, best_cost, eval_count
        eval_count += 1
        if eval_count > max_evaluations:
            return

        k = len(current_path)
        if k == length:
            consider_full_path(current_path)
            return

        # Prune if the bare minimum remaining edge + node cost already exceeds best_cost
        if current_edge_cost >= best_cost:
            return

        u = current_path[-1]
        # Greedy-first: sort unvisited neighbors by edge weight + candidate node readout & gate cost
        unvisited_nbrs = [nbr for nbr in graph.neighbors(u) if nbr not in current_path]
        unvisited_nbrs.sort(
            key=lambda v: (
                graph[u][v].get("weight", 0.002)
                + graph.nodes[v].get("ro_cost", 0.01)
                + graph.nodes[v].get("sq_cost", 0.0004)
            )
        )

        for v in unvisited_nbrs:
            w = graph[u][v].get("weight", 0.002)
            cz_mult = coupler_cz_counts[k - 1] if (k - 1) < len(coupler_cz_counts) else 4.0
            step_cost = cz_mult * w + graph.nodes[v].get("ro_cost", 0.01)

            if current_edge_cost + step_cost < best_cost:
                current_path.append(v)
                dfs(current_path, current_edge_cost + step_cost)
                current_path.pop()

    for start_node in candidate_starts[:25]:
        q0_init_cost = graph.nodes[start_node].get("ro_cost", 0.01) + graph.nodes[start_node].get("idle_cost_rate", 0.001) * (length - 1)
        dfs([start_node], q0_init_cost)
        if eval_count > max_evaluations:
            break

    if best_path is None:
        # Fall back to finding connected components and extracting a path
        for comp in nx.connected_components(graph):
            if len(comp) >= length:
                subg = graph.subgraph(comp)
                tree = nx.minimum_spanning_tree(subg)
                walk = list(nx.dfs_preorder_nodes(tree))
                candidate = walk[:length]
                # Compare forward vs reverse
                cost_fwd = _evaluate_directed_chain(
                    candidate, graph, coupler_cz_counts, sq_gate_counts, idle_weights, readout_line_penalty_weight
                )
                cost_rev = _evaluate_directed_chain(
                    list(reversed(candidate)), graph, coupler_cz_counts, sq_gate_counts, idle_weights, readout_line_penalty_weight
                )
                return candidate if cost_fwd <= cost_rev else list(reversed(candidate))
        raise ValueError(f"Could not find connected subgraph of size {length}")

    return best_path


def check_readout_line_constraint(qubits_measured: Iterable[cirq.GridQubit], all_active_qubits: Iterable[cirq.GridQubit]) -> bool:
    """Validate that mid-circuit measurements do not dephase active unmeasured qubits.

    On Willow, qubits sharing a diagonal line (row + col = const) share a readout line.
    Measuring some but not all qubits on that readout line dephases unmeasured qubits.
    """
    measured_set = set(qubits_measured)
    active_set = set(all_active_qubits)

    diagonal_groups: dict[int, list[cirq.GridQubit]] = {}
    for q in active_set:
        diag = q.row + q.col
        diagonal_groups.setdefault(diag, []).append(q)

    for diag, group in diagonal_groups.items():
        group_measured = [q for q in group if q in measured_set]
        if 0 < len(group_measured) < len(group):
            # Partial measurement on shared line induces dephasing
            return False
    return True
