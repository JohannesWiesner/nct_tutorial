"""View helpers for Cytoscape styles, Dash controls, and Plotly figures.

Functions here consume UI data or computed model results. They never run
control solvers or compute structural-connectivity metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc

from .graph_utils import split_elements
from .time_utils import CONTINUOUS_TIME_STEP, validate_time_system


def stylesheet_for(elements: dict) -> list[dict]:
    """Build Cytoscape styles for the current graph.

    Node borders indicate controller/constraint flags. Edge widths scale
    with the current weight range, including empty or equal-weight networks.
    """
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
    solution; each figure contains only that method's trajectories.
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
    figure.update_xaxes(title_text="Time step" if system == "discrete" else "Time (model units)")
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
    *, system: str | None = None,
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
        title="Aggregate control energy by node" + (f" ({system})" if system else ""),
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
    """Build the time selector and numeric inputs with persistent labels.

    Labels remain visible when inputs contain values, and their htmlFor
    attributes associate them with the matching fields. Horizon controls
    describe steps or durations according to the selected time system.
    """
    system = parameters.get("system", "discrete")
    horizon_label, minimum, step, help_text = horizon_control_settings(system)
    specs = (
        ("c", "Normalization (c)", 0.000001, "any"),
        ("T", horizon_label, minimum, step),
        ("rho", "Control weight (rho)", 0.000001, "any"),
    )
    numeric_controls = dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(
                        label, htmlFor=name, className="form-label",
                        **({"id": "T-label"} if name == "T" else {}),
                    ),
                    dbc.Input(
                        id=name, type="number", min=minimum, step=step,
                        value=parameters[name],
                    ),
                ],
                xs=12, sm=4,
            )
            for name, label, minimum, step in specs
        ],
        className="g-2",
    )
    return html.Div(
        [
            html.Label("Time system", htmlFor="time-system", className="form-label"),
            dbc.Select(
                id="time-system", value=system,
                options=[
                    {"label": "Discrete", "value": "discrete"},
                    {"label": "Continuous", "value": "continuous"},
                ],
                className="mb-2",
            ),
            numeric_controls,
            html.Small(help_text, id="time-horizon-help", className="text-muted"),
        ]
    )
