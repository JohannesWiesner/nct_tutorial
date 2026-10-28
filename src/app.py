"""Interactive Network Control Theory tutorial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import ClientsideFunction, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from .graph_utils import (
    add_node,
    decode_upload,
    default_store,
    delete_nodes,
    matrices_from_elements,
    merge_state_edits,
    node_ids,
    same_graph_data,
    states_from_rows,
    toggle_edges,
    toggle_node_flag,
    update_edge_weight,
)
from .nct_utils import compute_control_results, topology_metrics
from .view_utils import (
    control_figure,
    empty_control_figure,
    energy_figure,
    horizon_control_settings,
    metric_cards,
    parameter_controls,
    stylesheet_for,
    trajectory_figure,
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


initial = default_store(DEFAULT_ADJACENCY)
app = Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=str(Path(__file__).resolve().parent.parent / "assets"),
)
server = app.server
app.title = "Network Control Theory Tutorial"

app.layout = dbc.Container(
    [
        dcc.Store(id="network-store", data=initial),
        dcc.Store(id="viewport-guard"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        cyto.Cytoscape(
                            id="cytoscape",
                            layout={"name": "cose", "animate": False, "fit": True, "padding": 32},
                            elements=initial["elements"],
                            stylesheet=stylesheet_for(initial["elements"]),
                            style={
                                "width": "100%", "height": "520px",
                                "minHeight": "520px", "flex": "1 1 auto",
                                "border": "1px solid #ced4da", "boxSizing": "border-box",
                            },
                            maxZoom=4,
                            responsive=True,
                        ),
                        html.Div(
                            [
                                dbc.Button(
                                    "Fit graph", id="fit-graph", n_clicks=0,
                                    color="secondary", outline=True,
                                    title="Fit all nodes inside the canvas",
                                ),
                                dbc.Button(
                                    "Plot trajectories and energy", id="plot",
                                    color="primary", n_clicks=0,
                                ),
                            ],
                            className="d-flex flex-wrap gap-2 mt-2 flex-shrink-0",
                        ),
                        html.Div(id="plot-status", className="mt-2 text-danger"),
                    ],
                    lg=7, id="network-panel", className="d-flex flex-column",
                ),
                dbc.Col(
                    [
                        html.Section(
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
                                dbc.Button(
                                    "Switch x0 / xf", id="switch-states",
                                    className="mt-2", n_clicks=0,
                                ),
                            ],
                            className="border rounded p-3",
                        ),
                        html.Section(
                            [
                                html.H4("Parameters"),
                                parameter_controls(initial),
                            ],
                            className="border rounded p-3",
                        ),
                        html.Section(
                            [
                                html.H4("Edit network"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Add node", id="add-node", n_clicks=0),
                                        dbc.Button("Delete selected", id="delete-node", color="danger", n_clicks=0),
                                    ],
                                    className="mb-2",
                                ),
                                html.Div(
                                    [
                                        dbc.Button("Connect / disconnect", id="toggle-edge", n_clicks=0),
                                        dbc.Button("Toggle controller", id="toggle-controller", n_clicks=0),
                                        dbc.Button("Toggle constraint", id="toggle-constraint", n_clicks=0),
                                    ],
                                    className="mb-2 d-flex flex-wrap gap-2",
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
                            ],
                            className="border rounded p-3",
                        ),
                    ],
                    lg=5, id="network-controls", className="d-flex flex-column gap-3",
                ),
            ],
            className="g-3 align-items-stretch",
        ),
        html.Section(
            [
                html.H3("Topology measures"),
                dbc.Row(metric_cards(topology_metrics(DEFAULT_ADJACENCY)), id="topology-metrics", className="g-2"),
            ],
            className="border rounded p-3 mt-4",
        ),
        html.Section(
            [
                html.H3("Control Energy Measures"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="state-trajectory", figure=go.Figure()), lg=6),
                        dbc.Col(dcc.Graph(id="aggregate-energy", figure=go.Figure()), lg=6),
                        dbc.Col(dcc.Graph(
                            id="minimum-energy", figure=empty_control_figure("Minimum control energy trajectories"),
                        ), lg=6),
                        dbc.Col(dcc.Graph(
                            id="optimal-energy", figure=empty_control_figure("Optimal control energy trajectories"),
                        ), lg=6),
                    ],
                    className="g-0 mt-3",
                ),
            ],
            className="border rounded p-3 mt-4",
        ),
    ],
    fluid=True,
    className="py-3",
)


app.clientside_callback(
    ClientsideFunction(namespace="nctViewport", function_name="keepVisible"),
    Output("viewport-guard", "data"),
    Input("cytoscape", "extent"),
    Input("cytoscape", "elements"),
    Input("fit-graph", "n_clicks"),
)


@app.callback(
    Output("T-label", "children"),
    Output("T", "min"),
    Output("T", "step"),
    Output("time-horizon-help", "children"),
    Input("time-system", "value"),
)
def update_time_controls(system):
    """Keep the horizon label and input bounds aligned with the time system."""
    return horizon_control_settings(system)


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
    State("time-system", "value"),
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
    system,
):
    """Dispatch one user edit and keep graph data and state-table rows in sync."""
    action = ctx.triggered_id
    try:
        if action == "upload-data":
            return decode_upload(upload_contents), "Configuration loaded."

        store = merge_state_edits(dict(store), state_rows)
        store.update({"c": c, "T": time_horizon, "rho": rho, "system": system})
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
    Output("time-system", "value"),
    Input("network-store", "data"),
    State("cytoscape", "elements"),
    prevent_initial_call=True,
)
def render_network(store, current_elements):
    """Update graph props only when node/edge data changed.

    State swaps and parameter edits preserve Cytoscape's current positions,
    selection, and viewport by leaving its elements and stylesheet untouched.
    """
    elements = stylesheet = metrics = no_update
    if not same_graph_data(store["elements"], current_elements):
        elements = store["elements"]
        stylesheet = stylesheet_for(elements)
        adjacency, _, _, _ = matrices_from_elements(elements)
        metrics = metric_cards(topology_metrics(adjacency))
    return (
        elements,
        stylesheet,
        store["states"],
        store["c"],
        store["T"],
        store["rho"],
        metrics,
        store.get("system", "discrete"),
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
    State("time-system", "value"),
    prevent_initial_call=True,
)
def update_figures(_clicks, store, state_rows, c, time_horizon, rho, system):
    """Plot each method in its reserved figure using the selected time system."""
    try:
        adjacency, control, constraint, ids = matrices_from_elements(store["elements"])
        x0, xf = states_from_rows(state_rows, ids)
        result = compute_control_results(
            adjacency, control, constraint, x0, xf, time_horizon, c=c, rho=rho, system=system
        )
        return (
            trajectory_figure(
                result.uncontrolled, ids, f"Uncontrolled state trajectory ({system})",
                times=result.state_times, system=result.system,
            ),
            control_figure(
                result.minimum.states, result.minimum.inputs, ids,
                f"Minimum control energy trajectories ({system})",
                state_times=result.state_times, input_times=result.input_times, system=result.system,
            ),
            control_figure(
                result.optimal.states, result.optimal.inputs, ids,
                f"Optimal control energy trajectories ({system})",
                state_times=result.state_times, input_times=result.input_times, system=result.system,
            ),
            energy_figure(result.minimum.energy, result.optimal.energy, ids, system=system),
            "",
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return no_update, no_update, no_update, no_update, f"Could not compute figures: {error}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
