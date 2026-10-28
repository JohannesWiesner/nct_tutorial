"""Cytoscape editing, configuration parsing, and adapters to model arrays.

This module owns the UI data representation. It does not compute control
solutions or topology metrics, and it does not build figures.
"""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np

from .time_utils import validate_horizon, validate_time_system


def node_sort_key(node_id: str) -> tuple[int, int | str]:
    """Return a stable ordering key for a Cytoscape node ID.

    Numeric IDs sort numerically (2 before 10); other IDs sort lexically.
    """
    text = str(node_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def split_elements(elements: Mapping[str, list] | list) -> dict[str, list]:
    """Copy elements into explicit nodes and edges lists.

    Accept a grouped dictionary or a legacy flat list. Flat elements with
    source and target data are edges; all others are nodes. Returned data
    can be edited without modifying the caller's collection.
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
    """Return the network's node IDs in deterministic matrix/state order.

    Both grouped and legacy flat Cytoscape elements are accepted.
    """
    grouped = split_elements(elements)
    return sorted(
        (str(node["data"]["id"]) for node in grouped["nodes"]),
        key=node_sort_key,
    )


def same_graph_data(first: Mapping[str, list] | list, second: Mapping[str, list] | list) -> bool:
    """Compare node/edge data while ignoring positions, selection, and ordering.

    Cytoscape owns the current visual layout. State-table or parameter
    edits should not resend stored coordinates and restart that layout.
    IDs, weights, and controller/constraint flags still count as changes.
    """
    def data_by_id(elements):
        return {
            group: {str(element["data"]["id"]): element["data"] for element in entries}
            for group, entries in split_elements(elements).items()
        }

    return data_by_id(first) == data_by_id(second)


def from_adjacency_matrix(
    adjacency: np.ndarray,
    controller: Iterable[bool] | None = None,
    constrain: Iterable[bool] | None = None,
) -> dict[str, list]:
    """Convert an undirected adjacency matrix into Cytoscape elements.

    A must be a finite symmetric N-by-N matrix. Optional controller and
    constrain sequences provide one Boolean flag per node and default to
    True. Nodes receive string IDs and circular starting positions; each
    nonzero off-diagonal connection becomes one weighted edge.
    """
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
    """Read a diagonal B or S matrix as one Boolean flag per node.

    None enables every node. Reject wrong dimensions, nonfinite values,
    and off-diagonal entries; nonzero diagonal entries mean enabled.
    """
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
    """Validate uploaded model data and build a JSON-safe UI store.

    A is required; B/S default to identity, x0/xf to zero, and c/T/rho to
    1/3/1. system defaults to discrete; continuous durations use increments
    of 0.001. Return grouped Cytoscape elements, state-table rows, and scalar
    parameters. Invalid shapes, states, or parameter values raise ValueError.
    """
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

    system = validate_time_system(payload.get("system", "discrete"))
    time_horizon = validate_horizon(payload.get("T", 3), system)

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
        "system": system,
    }


def next_node_id(elements: Mapping[str, list] | list) -> str:
    """Find the first unused nonnegative numeric ID.

    Existing IDs are left intact, so deleting a node does not renumber
    any other node and a later addition may reuse the vacant ID.
    """
    existing = set(node_ids(elements))
    candidate = 0
    while str(candidate) in existing:
        candidate += 1
    return str(candidate)


def add_node(elements: Mapping[str, list] | list) -> tuple[dict[str, list], str]:
    """Return a copied network with one new controller/constrained node.

    The second return value is its string ID; callers use it to add a
    corresponding initial/target-state row to the UI store.
    """
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
    """Remove selected nodes and their incident edges from a copy.

    IDs and attributes of surviving nodes are preserved. State-table rows
    are separate UI data and must be removed by the caller.
    """
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
    """Toggle controller or constrain for the selected node IDs.

    Return grouped elements without mutating the input. Unknown flag
    names raise ValueError; IDs absent from the network are ignored.
    """
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
    """Set a positive finite weight on selected undirected edges.

    selected_edges contains Cytoscape edge data dictionaries, not full
    elements. Source/target order is ignored; unselected edges are unchanged.
    """
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
    """Connect selected nodes, or remove selected undirected edges.

    Selecting nodes creates all missing pairwise connections with the given
    positive weight. Selecting edges removes them. Mixed selections and
    fewer than two nodes without any edges raise ValueError.
    """
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
    """Translate Cytoscape elements into model arrays A, B, S and ordered IDs.

    A contains symmetric edge weights; B and S are diagonal 0/1 flags.
    All arrays use the returned ID order, including networks whose numeric
    IDs have gaps. Edges referring to missing nodes are ignored.
    """
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


def default_store(adjacency: np.ndarray) -> dict:
    """Create reproducible initial UI data for an adjacency matrix.

    Seeded initial/target values are stored by node ID alongside grouped
    elements and the default c/T/rho parameters and discrete time system.
    """
    rng = np.random.default_rng(28)
    ids = [str(index) for index in range(len(adjacency))]
    return {
        "elements": from_adjacency_matrix(adjacency),
        "states": [
            {"id": node_id, "x0": round(float(x0), 2), "xf": round(float(xf), 2)}
            for node_id, x0, xf in zip(ids, rng.random(len(ids)), rng.random(len(ids)))
        ],
        "c": 1.0,
        "T": 3,
        "rho": 1.0,
        "system": "discrete",
    }


def decode_upload(contents: str) -> dict:
    """Decode the base64 data URL emitted by Dash's upload component.

    Parse a UTF-8 JSON object and validate it as a network configuration.
    Malformed encoding or JSON raises a user-readable ValueError.
    """
    if not contents or "," not in contents:
        raise ValueError("Choose a JSON configuration file")
    _, encoded = contents.split(",", 1)
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The upload is not valid UTF-8 JSON") from error
    return validate_configuration(payload)


def merge_state_edits(store: dict, rows: list[dict] | None) -> dict:
    """Apply edited state-table rows to a UI store.

    Keep only IDs present in the current network and convert both state
    values to floats. The supplied store is updated and returned.
    """
    if rows is None:
        return store
    valid_ids = set(node_ids(store["elements"]))
    store["states"] = [
        {"id": str(row["id"]), "x0": float(row["x0"]), "xf": float(row["xf"])}
        for row in rows
        if str(row.get("id")) in valid_ids
    ]
    return store


def states_from_rows(rows: list[dict], ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read initial and target vectors in the model's node order.

    State-table ordering may differ after edits or node deletion. Match by
    ID and reject missing rows rather than silently associating wrong states.
    """
    if not ids:
        raise ValueError("The network has no nodes")
    by_id = {str(row["id"]): row for row in rows}
    if any(node_id not in by_id for node_id in ids):
        raise ValueError("State values are missing for one or more nodes")
    return (
        np.asarray([by_id[node_id]["x0"] for node_id in ids], dtype=float),
        np.asarray([by_id[node_id]["xf"] for node_id in ids], dtype=float),
    )
