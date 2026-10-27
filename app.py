"""Interactive Network Control Theory tutorial."""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ClientsideFunction, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from network_control.energies import minimum_input, optimal_input
from network_control.utils import matrix_normalization

from nct_utils import (
    add_node,
    aggregate_control_energy,
    delete_nodes,
    from_adjacency_matrix,
    matrices_from_elements,
    node_ids,
    split_elements,
    state_trajectory,
    toggle_edges,
    toggle_node_flag,
    topology_metrics,
    update_edge_weight,
    validate_configuration,
)


DEFAULT_ADJACENCY = np.array(
    [
        [0, 1, 2, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 3, 0, 0, 0, 0, 0],
        [2, 0, 0, 4, 0, 0, 0, 0, 0],
        [1, 3, 4, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 6, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
    ],
    dtype=float,
)


def default_store() -> dict:
    rng = np.random.default_rng(28)
    ids = [str(index) for index in range(len(DEFAULT_ADJACENCY))]
    return {
        "elements": from_adjacency_matrix(DEFAULT_ADJACENCY),
        "states": [
            {"id": node_id, "x0": round(float(x0), 2), "xf": round(float(xf), 2)}
            for node_id, x0, xf in zip(ids, rng.random(len(ids)), rng.random(len(ids)))
        ],
        "c": 1.0,
        "T": 3,
        "rho": 1.0,
    }


def stylesheet_for(elements: dict) -> list[dict]:
    grouped = split_elements(elements)
    weights = [float(edge["data"].get("weight", 1)) for edge in grouped["edges"]]
    low = min(weights, default=1.0)
    high = max(weights, default=1.0)
    if np.isclose(low, high):
        high = low + 1.0
    return [
        {"selector": "node", "style": {"content": "data(label)"}},
        {
            "selector": "[?controller][?constrain]",
            "style": {"border-width": 2, "border-color": "black", "border-style": "solid"},
        },
        {
            "selector": "[?controller][!constrain]",
            "style": {"border-width": 2, "border-color": "black", "border-style": "dashed"},
        },
        {
            "selector": "[!controller][?constrain]",
            "style": {"border-width": 2, "border-color": "black", "border-style": "dotted"},
        },
        {"selector": "[!controller][!constrain]", "style": {}},
        {
            "selector": "edge",
            "style": {"width": f"mapData(weight,{low},{high},1,6)", "label": "data(weight)"},
        },
    ]


def metric_cards(elements: dict) -> list:
    labels = {
        "nodes": "Nodes",
        "edges": "Edges",
        "density": "Density",
        "mean_degree": "Mean degree",
        "mean_strength": "Mean strength",
        "mean_clustering": "Mean clustering",
    }
    metrics = topology_metrics(elements)
    return [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Small(labels[key], className="text-muted"),
                        html.H5(value if isinstance(value, int) else f"{value:.3f}"),
                    ]
                )
            ),
            xs=6,
            md=2,
        )
        for key, value in metrics.items()
    ]


def decode_upload(contents: str) -> dict:
    if not contents or "," not in contents:
        raise ValueError("Choose a JSON configuration file")
    _, encoded = contents.split(",", 1)
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The upload is not valid UTF-8 JSON") from error
    return validate_configuration(payload)


def merge_state_edits(store: dict, rows: list[dict] | None) -> dict:
    if rows is None:
        return store
    valid_ids = set(node_ids(store["elements"]))
    store["states"] = [
        {"id": str(row["id"]), "x0": float(row["x0"]), "xf": float(row["xf"])}
        for row in rows
        if str(row.get("id")) in valid_ids
    ]
    return store


def trajectory_figure(trajectory: np.ndarray, ids: list[str], title: str) -> go.Figure:
    frame = pd.DataFrame(trajectory, columns=ids)
    frame.index.name = "t"
    tidy = frame.reset_index().melt(id_vars="t", var_name="Node", value_name="Value")
    figure = px.line(tidy, x="t", y="Value", color="Node", title=title)
    figure.update_layout(title_x=0.5)
    return figure


def control_figure(states: np.ndarray, control: np.ndarray, ids: list[str], title: str) -> go.Figure:
    frames = []
    for values, kind in ((states, "State x"), (control, "Input u")):
        columns = ids[: values.shape[1]]
        frame = pd.DataFrame(values[:, : len(columns)], columns=columns)
        frame.index.name = "t"
        tidy = frame.reset_index().melt(id_vars="t", var_name="Node", value_name="Value")
        tidy["Type"] = kind
        frames.append(tidy)
    figure = px.line(
        pd.concat(frames, ignore_index=True),
        x="t",
        y="Value",
        color="Node",
        facet_row="Type",
        title=title,
    )
    figure.update_layout(title_x=0.5)
    return figure


def energy_figure(minimum_u: np.ndarray, optimal_u: np.ndarray, ids: list[str]) -> go.Figure:
    minimum = aggregate_control_energy(minimum_u)
    optimal = aggregate_control_energy(optimal_u)
    count = min(len(ids), len(minimum), len(optimal))
    frame = pd.DataFrame(
        {
            "Node": ids[:count] * 2,
            "Energy": np.concatenate((minimum[:count], optimal[:count])),
            "Method": ["Minimum"] * count + ["Optimal"] * count,
        }
    )
    figure = px.bar(
        frame,
        x="Node",
        y="Energy",
        color="Method",
        barmode="group",
        title="Aggregate control energy by node",
    )
    figure.update_layout(title_x=0.5)
    return figure


initial = default_store()
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Network Control Theory Tutorial"

app.layout = dbc.Container(
    [
        dcc.Store(id="network-store", data=initial),
        dcc.Store(id="viewport-guard"),
        html.H1("Network Control Theory Tutorial", className="my-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        cyto.Cytoscape(
                            id="cytoscape",
                            layout={"name": "cose", "animate": False, "fit": True, "padding": 32},
                            elements=initial["elements"],
                            stylesheet=stylesheet_for(initial["elements"]),
                            style={"width": "100%", "height": "520px"},
                            maxZoom=4,
                            responsive=True,
                        ),
                        dbc.Button(
                            "Fit graph", id="fit-graph", n_clicks=0,
                            color="secondary", outline=True, className="mt-2",
                            title="Fit all nodes inside the canvas",
                        ),
                    ],
                    lg=7,
                ),
                dbc.Col(
                    [
                        html.H4("States"),
                        dash_table.DataTable(
                            id="states-table",
                            columns=[
                                {"id": "id", "name": "Node", "editable": False},
                                {"id": "x0", "name": "x0", "type": "numeric"},
                                {"id": "xf", "name": "xf", "type": "numeric"},
                            ],
                            data=initial["states"],
                            editable=True,
                        ),
                        dbc.Button("Switch x0 / xf", id="switch-states", className="my-2", n_clicks=0),
                        html.H4("Parameters", className="mt-3"),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="c", type="number", min=0.000001, value=initial["c"], placeholder="c")),
                                dbc.Col(dbc.Input(id="T", type="number", min=1, step=1, value=initial["T"], placeholder="T")),
                                dbc.Col(dbc.Input(id="rho", type="number", min=0.000001, value=initial["rho"], placeholder="rho")),
                            ],
                            className="g-2",
                        ),
                        html.H4("Edit network", className="mt-3"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Add node", id="add-node", n_clicks=0),
                                dbc.Button("Delete selected", id="delete-node", color="danger", n_clicks=0),
                            ],
                            className="mb-2",
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Connect / disconnect", id="toggle-edge", n_clicks=0),
                                dbc.Button("Toggle controller", id="toggle-controller", n_clicks=0),
                                dbc.Button("Toggle constraint", id="toggle-constraint", n_clicks=0),
                            ],
                            className="mb-2 d-flex",
                        ),
                        dbc.InputGroup(
                            [
                                dbc.Input(id="edge-weight", type="number", min=0.000001, value=1),
                                dbc.Button("Set selected edge weight", id="set-edge-weight", n_clicks=0),
                            ],
                            className="mb-3",
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=dbc.Button("Upload JSON configuration", color="secondary"),
                            accept="application/json,.json",
                        ),
                        html.Div(id="edit-status", className="mt-2"),
                        dbc.Button("Plot trajectories and energy", id="plot", color="primary", className="mt-3", n_clicks=0),
                        html.Div(id="plot-status", className="mt-2 text-danger"),
                    ],
                    lg=5,
                ),
            ],
            className="g-3",
        ),
        html.H3("Topology measures", className="mt-4"),
        dbc.Row(metric_cards(initial["elements"]), id="topology-metrics", className="g-2"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="state-trajectory", figure=go.Figure()), lg=6),
                dbc.Col(dcc.Graph(id="aggregate-energy", figure=go.Figure()), lg=6),
                dbc.Col(dcc.Graph(id="minimum-energy", figure=go.Figure()), lg=6),
                dbc.Col(dcc.Graph(id="optimal-energy", figure=go.Figure()), lg=6),
            ],
            className="g-0 mt-3",
        ),
    ],
    fluid=True,
)


app.clientside_callback(
    ClientsideFunction(namespace="nctViewport", function_name="keepVisible"),
    Output("viewport-guard", "data"),
    Input("cytoscape", "extent"),
    Input("cytoscape", "elements"),
    Input("fit-graph", "n_clicks"),
)


@app.callback(
    Output("network-store", "data"),
    Output("edit-status", "children"),
    Input("add-node", "n_clicks"),
    Input("delete-node", "n_clicks"),
    Input("toggle-edge", "n_clicks"),
    Input("toggle-controller", "n_clicks"),
    Input("toggle-constraint", "n_clicks"),
    Input("set-edge-weight", "n_clicks"),
    Input("switch-states", "n_clicks"),
    Input("upload-data", "contents"),
    State("network-store", "data"),
    State("states-table", "data"),
    State("cytoscape", "selectedNodeData"),
    State("cytoscape", "selectedEdgeData"),
    State("edge-weight", "value"),
    State("c", "value"),
    State("T", "value"),
    State("rho", "value"),
    prevent_initial_call=True,
)
def update_network(
    _add_clicks,
    _delete_clicks,
    _edge_clicks,
    _controller_clicks,
    _constraint_clicks,
    _weight_clicks,
    _switch_clicks,
    upload_contents,
    store,
    state_rows,
    selected_nodes,
    selected_edges,
    edge_weight,
    c,
    time_horizon,
    rho,
):
    action = ctx.triggered_id
    try:
        if action == "upload-data":
            return decode_upload(upload_contents), "Configuration loaded."

        store = merge_state_edits(dict(store), state_rows)
        store.update({"c": c, "T": time_horizon, "rho": rho})
        selected_nodes = selected_nodes or []
        selected_edges = selected_edges or []
        selected = [str(node["id"]) for node in selected_nodes]

        if action == "add-node":
            store["elements"], new_id = add_node(store["elements"])
            store["states"].append({"id": new_id, "x0": 0.0, "xf": 0.0})
            message = f"Added node {new_id}."
        elif action == "delete-node":
            if not selected:
                raise ValueError("Select at least one node to delete")
            if len(selected) >= len(node_ids(store["elements"])):
                raise ValueError("The network must contain at least one node")
            store["elements"] = delete_nodes(store["elements"], selected)
            store["states"] = [row for row in store["states"] if str(row["id"]) not in set(selected)]
            message = f"Deleted {len(selected)} node(s); remaining IDs were preserved."
        elif action == "toggle-edge":
            store["elements"] = toggle_edges(store["elements"], selected_nodes, selected_edges, edge_weight)
            message = "Updated the selected connections."
        elif action == "toggle-controller":
            if not selected:
                raise ValueError("Select at least one node")
            store["elements"] = toggle_node_flag(store["elements"], selected, "controller")
            message = "Updated controller nodes."
        elif action == "toggle-constraint":
            if not selected:
                raise ValueError("Select at least one node")
            store["elements"] = toggle_node_flag(store["elements"], selected, "constrain")
            message = "Updated constrained nodes."
        elif action == "set-edge-weight":
            if not selected_edges:
                raise ValueError("Select at least one edge")
            store["elements"] = update_edge_weight(store["elements"], selected_edges, edge_weight)
            message = "Updated edge weights."
        elif action == "switch-states":
            for row in store["states"]:
                row["x0"], row["xf"] = row["xf"], row["x0"]
            message = "Switched x0 and xf."
        else:
            return no_update, no_update
        return store, message
    except (KeyError, TypeError, ValueError) as error:
        return no_update, f"Could not update network: {error}"


@app.callback(
    Output("cytoscape", "elements"),
    Output("cytoscape", "stylesheet"),
    Output("states-table", "data"),
    Output("c", "value"),
    Output("T", "value"),
    Output("rho", "value"),
    Output("topology-metrics", "children"),
    Input("network-store", "data"),
    prevent_initial_call=True,
)
def render_network(store):
    return (
        store["elements"],
        stylesheet_for(store["elements"]),
        store["states"],
        store["c"],
        store["T"],
        store["rho"],
        metric_cards(store["elements"]),
    )


@app.callback(
    Output("state-trajectory", "figure"),
    Output("minimum-energy", "figure"),
    Output("optimal-energy", "figure"),
    Output("aggregate-energy", "figure"),
    Output("plot-status", "children"),
    Input("plot", "n_clicks"),
    State("network-store", "data"),
    State("states-table", "data"),
    State("c", "value"),
    State("T", "value"),
    State("rho", "value"),
    prevent_initial_call=True,
)
def update_figures(_clicks, store, state_rows, c, time_horizon, rho):
    try:
        if not isinstance(time_horizon, int) or isinstance(time_horizon, bool) or time_horizon < 1:
            raise ValueError("T must be a positive integer")
        if c is None or float(c) <= 0 or rho is None or float(rho) <= 0:
            raise ValueError("c and rho must be positive")

        adjacency, control, constraint, ids = matrices_from_elements(store["elements"])
        if not ids:
            raise ValueError("The network has no nodes")
        rows = {str(row["id"]): row for row in state_rows}
        if any(node_id not in rows for node_id in ids):
            raise ValueError("State values are missing for one or more nodes")
        x0 = np.asarray([rows[node_id]["x0"] for node_id in ids], dtype=float).reshape(-1, 1)
        xf = np.asarray([rows[node_id]["xf"] for node_id in ids], dtype=float).reshape(-1, 1)
        normalized = matrix_normalization(adjacency, float(c))

        uncontrolled = state_trajectory(normalized, x0, time_horizon)
        minimum_x, minimum_u, _minimum_error = minimum_input(
            normalized, time_horizon, control, x0, xf
        )
        optimal_x, optimal_u, _optimal_error = optimal_input(
            normalized, time_horizon, control, x0, xf, float(rho), constraint
        )
        return (
            trajectory_figure(uncontrolled, ids, "Uncontrolled state trajectory"),
            control_figure(minimum_x, minimum_u, ids, "Minimum control energy"),
            control_figure(optimal_x, optimal_u, ids, "Optimal control energy"),
            energy_figure(minimum_u, optimal_u, ids),
            "",
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return no_update, no_update, no_update, no_update, f"Could not compute figures: {error}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
