"""View helpers for Cytoscape styles, Dash controls, and Plotly figures.

Cytoscape styles, persistent parameter labels, topology cards, and Plotly figures. Consumes precomputed metrics, energies, and time coordinates.

Functions here consume UI data or computed model results. They never run
control solvers or compute structural-connectivity metrics.
"""

from __future__ import annotations

from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc

from .graph_utils import split_elements
from .time_utils import CONTINUOUS_TIME_STEP, validate_time_system


# The same SVGs are used on the canvas and in the legend. Opaque discs keep
# crossing edges out of the icons; shapes and fixed positions also encode roles.
def _badge_svg(symbol: str, color: str) -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">'
        '<circle cx="10" cy="10" r="9.25" fill="white" stroke="#8793a1" stroke-width="1.5"/>'
        f'<g fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round">{symbol}</g></svg>'
    )
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")


CONTROL_BADGE = _badge_svg('<path d="M11 3.5 5.5 11H10l-1 5.5 5.5-8H10Z"/>', "#2364ad")
PENALTY_BADGE = _badge_svg(
    '<circle cx="10" cy="10" r="5.5"/><circle cx="10" cy="10" r="2.5"/>'
    '<circle cx="10" cy="10" r="0.75" fill="#a8540b" stroke="none"/>',
    "#a8540b",
)


def stylesheet_for(elements: dict) -> list[dict]:
    """Build Cytoscape styles for the current graph.

    Fixed rim badges indicate B/S roles without changing node geometry.
    Edge widths encode weights; edges have no text labels.
    """
    grouped = split_elements(elements)
    weights = [float(edge["data"].get("weight", 1)) for edge in grouped["edges"]]
    low = min(weights, default=1.0)
    high = max(weights, default=1.0)
    if np.isclose(low, high):
        high = low + 1.0
    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)", "width": 48, "height": 48,
                "font-size": 16, "font-weight": 600, "color": "#182230",
                "text-valign": "center", "text-halign": "center",
                "background-color": "#f4f6f9", "border-width": 1.5,
                "border-color": "#8793a1", "border-style": "solid",
                "background-image": [CONTROL_BADGE, PENALTY_BADGE],
                "background-width": [20, 20], "background-height": [20, 20],
                "background-position-x": ["0%", "100%"],
                "background-position-y": ["0%", "0%"],
                "background-offset-x": [-5, 5], "background-offset-y": [-7, -7],
                "background-clip": ["none", "none"],
                "background-image-containment": ["over", "over"],
                "background-image-opacity": [0, 0],
                # Reserve badge and selection extents even when roles are off.
                "bounds-expansion": 10,
            },
        },
        {
            "selector": "node[?controller][?constrain]",
            "style": {"background-image-opacity": [1, 1]},
        },
        {
            "selector": "node[?controller][!constrain]",
            "style": {"background-image-opacity": [1, 0]},
        },
        {
            "selector": "node[!controller][?constrain]",
            "style": {"background-image-opacity": [0, 1]},
        },
        {
            "selector": "node:selected",
            "style": {
                "border-color": "#6d28d9", "underlay-color": "#6d28d9",
                "underlay-opacity": 0.25, "underlay-padding": 5, "underlay-shape": "ellipse",
            },
        },
        {
            "selector": "edge",
            "style": {"width": f"mapData(weight,{low},{high},1,6)", "label": ""},
        },
    ]


def node_role_legend() -> html.Div:
    """Keep role meanings visible without requiring color or hover tooltips."""
    return html.Div(
        [
            html.Div([
                html.Span([html.Img(src=CONTROL_BADGE, alt="", width=20, height=20),
                           "Control input (B)"], className="node-role-key"),
                html.Span([html.Img(src=PENALTY_BADGE, alt="", width=20, height=20),
                           "State penalty (S)"], className="node-role-key"),
            ], className="d-flex flex-wrap gap-3"),
            html.Small("S weights the state-cost term in optimal control.", className="text-muted"),
        ],
        className="node-role-legend mt-3", role="note", **{"aria-label": "Node roles"},
    )


def metric_cards(metrics: dict) -> list:
    """Render already-computed topology metrics as Bootstrap cards.

    The input maps metric names to scalar values from the numerical model;
    this function only supplies labels and display formatting.
    """
    labels = {
        "nodes": "Nodes",
        "edges": "Edges",
        "density": "Density",
        "mean_degree": "Mean degree",
        "mean_strength": "Mean strength",
        "mean_clustering": "Mean clustering",
    }
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


def trajectory_figure(
    trajectory: np.ndarray, ids: list[str], title: str,
    *, times: np.ndarray | None = None, system: str = "discrete",
) -> go.Figure:
    """Plot a time-by-node state array with one line per node.

    ids labels the array columns. times supplies model times, including
    fractional continuous times; omitted times default to row indices.
    """
    frame = pd.DataFrame(trajectory, columns=ids, index=times)
    frame.index.name = "t"
    tidy = frame.reset_index().melt(id_vars="t", var_name="Node", value_name="Value")
    figure = px.line(tidy, x="t", y="Value", color="Node", title=title)
    figure.update_layout(title_x=0.5)
    figure.update_xaxes(title_text="Time step" if system == "discrete" else "Time (model units)")
    return figure


def control_figure(
    states: np.ndarray, control: np.ndarray, ids: list[str], title: str,
    *, state_times: np.ndarray | None = None, input_times: np.ndarray | None = None,
    system: str = "discrete",
) -> go.Figure:
    """Plot one control method's states and inputs in separate panels.

    Both arrays have nodes in columns and time in rows. They may have
    different row counts: T control inputs connect T+1 discrete states.
    Explicit time arrays keep continuous sample indices out of the axes.
    The caller supplies either the minimum-energy or optimal-control
    solution; each figure contains only that method's trajectories,
    with a shared time-axis title beneath the bottom panel.
    """
    frames = []
    for values, kind, times in ((states, "State x", state_times), (control, "Input u", input_times)):
        columns = ids[: values.shape[1]]
        frame = pd.DataFrame(values[:, : len(columns)], columns=columns, index=times)
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
    figure.update_xaxes(title_text=None)
    figure.update_xaxes(
        title_text="Time step" if system == "discrete" else "Time (model units)",
        row=1, col=1,
    )
    return figure


def empty_control_figure(title: str) -> go.Figure:
    """Label a control method's reserved plot before its first computation."""
    figure = go.Figure()
    figure.update_layout(title=title, title_x=0.5)
    figure.add_annotation(
        text="Click Plot trajectories and energy to compute this method's trajectories.",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def energy_figure(
    minimum_energy: np.ndarray, optimal_energy: np.ndarray, ids: list[str],
) -> go.Figure:
    """Compare already-computed minimum and optimal energy per node.

    Each energy vector follows ids. Numerical integration or summation is
    the model's responsibility; this function only constructs the bar chart.
    """
    minimum = np.asarray(minimum_energy, dtype=float)
    optimal = np.asarray(optimal_energy, dtype=float)
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


def horizon_control_settings(system: str) -> tuple[str, float, float, str]:
    """Return the horizon label, minimum, increment, and help for a system."""
    validate_time_system(system)
    if system == "continuous":
        return (
            "Time duration (T)", CONTINUOUS_TIME_STEP, CONTINUOUS_TIME_STEP,
            "Positive duration in model time units, in increments of 0.001.",
        )
    return "Time steps (T)", 1, 1, "Positive integer number of transitions."


def parameter_controls(parameters: dict) -> html.Div:
    """Build responsive, aligned time-system, T, c, and rho controls.

    Labels remain visible when inputs contain values, and their htmlFor
    attributes associate them with the matching fields. Shared grid rows
    keep inputs level when labels wrap, with help beneath each input.
    """
    system = parameters.get("system", "discrete")
    horizon_label, minimum, step, help_text = horizon_control_settings(system)
    specs = (
        ("T", horizon_label, minimum, step),
        ("c", "Normalization (c)", 0.000001, "any"),
        ("rho", "Control weight (rho)", 0.000001, "any"),
    )
    fields = [
        (
            "time-system", "Time system",
            dbc.Select(
                id="time-system", value=system,
                options=[
                    {"label": "Discrete", "value": "discrete"},
                    {"label": "Continuous", "value": "continuous"},
                ],
            ),
        ),
        *[
            (
                name, label,
                dbc.Input(
                    id=name, type="number", min=minimum, step=step,
                    value=parameters[name],
                ),
            )
            for name, label, minimum, step in specs
        ],
    ]
    descriptions = {
        "time-system": ("time-system-help", "Choose continuous dynamics or discrete time steps."),
        "T": ("time-horizon-help", help_text),
        "c": ("normalization-help", "Positive constant that stabilizes the network dynamics."),
        "rho": ("control-weight-help", "Higher values penalize control effort more in optimal control."),
    }
    children = []
    for name, label, control in fields:
        help_id, description = descriptions[name]
        children.append(html.Div(
            [
                html.Label(
                    label, htmlFor=name, className="form-label mb-0",
                    **({"id": "T-label"} if name == "T" else {}),
                ),
                control,
                html.Small(description, id=help_id, className="text-muted"),
            ],
            className="parameter-field",
        ))
    return html.Div(children, className="parameter-grid")
