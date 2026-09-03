"""Pure data and numerical helpers for the NCT tutorial application.

The Dash layer delegates network mutations to this module so they can be
tested without a browser (or even Dash) being installed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np


def node_sort_key(node_id: str) -> tuple[int, int | str]:
    """Sort numeric Cytoscape IDs numerically and other IDs lexically."""
    text = str(node_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def split_elements(elements: Mapping[str, list] | list) -> dict[str, list]:
    """Return elements as an explicit ``nodes``/``edges`` mapping.

    A flat list is accepted for backwards compatibility. Each element is
    inspected only once.
    """
    if isinstance(elements, Mapping):
        return {
            "nodes": deepcopy(list(elements.get("nodes", []))),
            "edges": deepcopy(list(elements.get("edges", []))),
        }

    grouped: dict[str, list] = {"nodes": [], "edges": []}
    for element in elements or []:
        data = element.get("data", {})
        group = "edges" if "source" in data and "target" in data else "nodes"
        grouped[group].append(deepcopy(element))
    return grouped


def node_ids(elements: Mapping[str, list] | list) -> list[str]:
    grouped = split_elements(elements)
    return sorted(
        (str(node["data"]["id"]) for node in grouped["nodes"]),
        key=node_sort_key,
    )


def from_adjacency_matrix(
    adjacency: np.ndarray,
    controller: Iterable[bool] | None = None,
    constrain: Iterable[bool] | None = None,
) -> dict[str, list]:
    """Convert a square adjacency matrix to explicit Cytoscape elements."""
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must contain only finite values")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("A must be symmetric for this undirected tutorial")

    size = matrix.shape[0]
    controllers = list(controller) if controller is not None else [True] * size
    constraints = list(constrain) if constrain is not None else [True] * size
    if len(controllers) != size or len(constraints) != size:
        raise ValueError("B and S must describe every node in A")

    nodes = []
    for index in range(size):
        angle = 2 * np.pi * index / max(size, 1)
        nodes.append(
            {
                "data": {
                    "id": str(index),
                    "label": str(index),
                    "controller": bool(controllers[index]),
                    "constrain": bool(constraints[index]),
                },
                "position": {
                    "x": float(220 + 180 * np.cos(angle)),
                    "y": float(220 + 180 * np.sin(angle)),
                },
            }
        )

    edges = []
    for source in range(size):
        for target in range(source + 1, size):
            weight = float(matrix[source, target])
            if weight != 0:
                edges.append(
                    {
                        "data": {
                            "id": f"{source}--{target}",
                            "source": str(source),
                            "target": str(target),
                            "weight": weight,
                        }
                    }
                )
    return {"nodes": nodes, "edges": edges}


def _diagonal_flags(matrix: Any, size: int, name: str) -> list[bool]:
    if matrix is None:
        return [True] * size
    value = np.asarray(matrix, dtype=float)
    if value.shape != (size, size):
        raise ValueError(f"{name} must have shape {size}x{size}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(value, np.diag(np.diag(value))):
        raise ValueError(f"{name} must be diagonal")
    return [bool(item) for item in np.diag(value)]


def validate_configuration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an uploaded NCT configuration and return JSON-safe values."""
    if not isinstance(payload, Mapping):
        raise ValueError("The uploaded JSON document must contain an object")
    if "A" not in payload:
        raise ValueError("The uploaded JSON document must include A")

    adjacency = np.asarray(payload["A"], dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("A must be a square matrix")
    size = adjacency.shape[0]
    controllers = _diagonal_flags(payload.get("B"), size, "B")
    constraints = _diagonal_flags(payload.get("S"), size, "S")

    def vector(name: str, fallback: np.ndarray) -> list[float]:
        value = np.asarray(payload.get(name, fallback), dtype=float).reshape(-1)
        if value.size != size or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain {size} finite values")
        return value.tolist()

    def positive_number(name: str, default: float) -> float:
        value = float(payload.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return value

    time_horizon = payload.get("T", 3)
    if isinstance(time_horizon, bool) or int(time_horizon) != float(time_horizon):
        raise ValueError("T must be a positive integer")
    time_horizon = int(time_horizon)
    if time_horizon < 1:
        raise ValueError("T must be a positive integer")

    x0_values = vector("x0", np.zeros(size))
    xf_values = vector("xf", np.zeros(size))
    return {
        "elements": from_adjacency_matrix(adjacency, controllers, constraints),
        "states": [
            {"id": str(index), "x0": x0, "xf": xf}
            for index, (x0, xf) in enumerate(zip(x0_values, xf_values))
        ],
        "c": positive_number("c", 1.0),
        "T": time_horizon,
        "rho": positive_number("rho", 1.0),
    }


def next_node_id(elements: Mapping[str, list] | list) -> str:
    existing = set(node_ids(elements))
    candidate = 0
    while str(candidate) in existing:
        candidate += 1
    return str(candidate)


def add_node(elements: Mapping[str, list] | list) -> tuple[dict[str, list], str]:
    grouped = split_elements(elements)
    new_id = next_node_id(grouped)
    grouped["nodes"].append(
        {
            "data": {
                "id": new_id,
                "label": new_id,
                "controller": True,
                "constrain": True,
            }
        }
    )
    return grouped, new_id


def delete_nodes(
    elements: Mapping[str, list] | list, selected_ids: Iterable[str]
) -> dict[str, list]:
    """Delete nodes and incident edges while preserving all other IDs."""
    grouped = split_elements(elements)
    selected = {str(value) for value in selected_ids}
    grouped["nodes"] = [
        node for node in grouped["nodes"] if str(node["data"]["id"]) not in selected
    ]
    grouped["edges"] = [
        edge
        for edge in grouped["edges"]
        if str(edge["data"]["source"]) not in selected
        and str(edge["data"]["target"]) not in selected
    ]
    return grouped


def toggle_node_flag(
    elements: Mapping[str, list] | list,
    selected_ids: Iterable[str],
    flag: str,
) -> dict[str, list]:
    if flag not in {"controller", "constrain"}:
        raise ValueError("Unknown node flag")
    grouped = split_elements(elements)
    selected = {str(value) for value in selected_ids}
    for node in grouped["nodes"]:
        if str(node["data"]["id"]) in selected:
            node["data"][flag] = not bool(node["data"].get(flag, True))
    return grouped


def update_edge_weight(
    elements: Mapping[str, list] | list,
    selected_edges: Iterable[Mapping[str, Any]],
    weight: float,
) -> dict[str, list]:
    grouped = split_elements(elements)
    numeric_weight = float(weight)
    if not np.isfinite(numeric_weight) or numeric_weight <= 0:
        raise ValueError("Edge weight must be a positive number")
    selected = {
        frozenset((str(edge["source"]), str(edge["target"])))
        for edge in selected_edges
    }
    for edge in grouped["edges"]:
        pair = frozenset((str(edge["data"]["source"]), str(edge["data"]["target"])))
        if pair in selected:
            edge["data"]["weight"] = numeric_weight
    return grouped


def toggle_edges(
    elements: Mapping[str, list] | list,
    selected_nodes: Iterable[Mapping[str, Any]],
    selected_edges: Iterable[Mapping[str, Any]],
    weight: float,
) -> dict[str, list]:
    grouped = split_elements(elements)
    nodes = [str(node["id"]) for node in selected_nodes]
    edges = list(selected_edges)
    if nodes and edges:
        raise ValueError("Select nodes or edges, not both")
    if edges:
        selected = {
            frozenset((str(edge["source"]), str(edge["target"]))) for edge in edges
        }
        grouped["edges"] = [
            edge
            for edge in grouped["edges"]
            if frozenset(
                (str(edge["data"]["source"]), str(edge["data"]["target"]))
            )
            not in selected
        ]
        return grouped
    if len(nodes) < 2:
        raise ValueError("Select at least two nodes to connect")

    numeric_weight = float(weight)
    if not np.isfinite(numeric_weight) or numeric_weight <= 0:
        raise ValueError("Edge weight must be a positive number")
    existing = {
        frozenset((str(edge["data"]["source"]), str(edge["data"]["target"])))
        for edge in grouped["edges"]
    }
    ordered = sorted(set(nodes), key=node_sort_key)
    for source_index, source in enumerate(ordered):
        for target in ordered[source_index + 1 :]:
            pair = frozenset((source, target))
            if pair not in existing:
                grouped["edges"].append(
                    {
                        "data": {
                            "id": f"{source}--{target}",
                            "source": source,
                            "target": target,
                            "weight": numeric_weight,
                        }
                    }
                )
                existing.add(pair)
    return grouped


def matrices_from_elements(
    elements: Mapping[str, list] | list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build A, B and S together from one normalized element collection."""
    grouped = split_elements(elements)
    ids = sorted(
        (str(node["data"]["id"]) for node in grouped["nodes"]), key=node_sort_key
    )
    positions = {node_id: index for index, node_id in enumerate(ids)}
    size = len(ids)
    adjacency = np.zeros((size, size), dtype=float)
    for edge in grouped["edges"]:
        data = edge["data"]
        source, target = str(data["source"]), str(data["target"])
        if source not in positions or target not in positions:
            continue
        i, j = positions[source], positions[target]
        adjacency[i, j] = adjacency[j, i] = float(data.get("weight", 1))

    node_by_id = {str(node["data"]["id"]): node["data"] for node in grouped["nodes"]}
    control = np.diag([float(bool(node_by_id[item].get("controller", True))) for item in ids])
    constraint = np.diag([float(bool(node_by_id[item].get("constrain", True))) for item in ids])
    return adjacency, control, constraint, ids


def topology_metrics(elements: Mapping[str, list] | list) -> dict[str, float | int]:
    """Compute compact undirected structural-connectivity measures."""
    adjacency, _, _, _ = matrices_from_elements(elements)
    size = adjacency.shape[0]
    binary = adjacency != 0
    edge_count = int(np.count_nonzero(np.triu(binary, 1)))
    density = 0.0 if size < 2 else 2 * edge_count / (size * (size - 1))
    degree = binary.sum(axis=1) if size else np.array([])
    strength = np.abs(adjacency).sum(axis=1) if size else np.array([])

    clustering_values = []
    for node in range(size):
        neighbors = np.flatnonzero(binary[node])
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        actual = np.count_nonzero(np.triu(binary[np.ix_(neighbors, neighbors)], 1))
        clustering_values.append(0.0 if possible == 0 else actual / possible)

    return {
        "nodes": size,
        "edges": edge_count,
        "density": float(density),
        "mean_degree": float(degree.mean()) if size else 0.0,
        "mean_strength": float(strength.mean()) if size else 0.0,
        "mean_clustering": float(np.mean(clustering_values)) if size else 0.0,
    }


def aggregate_control_energy(control_input: np.ndarray) -> np.ndarray:
    """Return discrete control energy per node, summed across time."""
    values = np.asarray(control_input, dtype=float)
    if values.ndim != 2:
        raise ValueError("Control input must be a two-dimensional array")
    return np.square(values).sum(axis=0)


def state_trajectory(
    adjacency: np.ndarray, initial_state: np.ndarray, time_horizon: int
) -> np.ndarray:
    """Simulate ``x(t + 1) = A x(t)`` for a positive integer horizon."""
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square matrix")
    if (
        isinstance(time_horizon, bool)
        or int(time_horizon) != time_horizon
        or time_horizon < 1
    ):
        raise ValueError("T must be a positive integer")
    state = np.asarray(initial_state, dtype=float).reshape(-1)
    if state.size != matrix.shape[0]:
        raise ValueError("The initial state must have one value per node")

    trajectory = np.zeros((int(time_horizon), matrix.shape[0]))
    for time_point in range(int(time_horizon)):
        trajectory[time_point] = state
        state = matrix @ state
    return trajectory
