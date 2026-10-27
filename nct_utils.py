"""Numerical model for the network control tutorial.

This module accepts arrays and returns arrays/scalars. nctpy supplies the
discrete-time control solver and normalization; bct supplies topology metrics.
It has no dependency on Dash, Plotly, or Cytoscape element dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import bct
import numpy as np
from nctpy.energies import get_control_inputs
from nctpy.utils import matrix_normalization


@dataclass(frozen=True)
class ControlSolution:
    """One control solution, independent of its eventual visualization.

    states has shape (T+1, N), inputs has shape (T, N), and energy contains
    N sums of squared inputs. errors retains the solver's two residuals.
    """

    states: np.ndarray
    inputs: np.ndarray
    energy: np.ndarray
    errors: tuple[float, float]


@dataclass(frozen=True)
class ControlResults:
    """Normalized dynamics and all numerical results needed by the tutorial.

    uncontrolled includes states from t=0 through T. Both control solutions
    use the same normalized adjacency matrix, time steps, and node ordering.
    """

    normalized_adjacency: np.ndarray
    uncontrolled: np.ndarray
    minimum: ControlSolution
    optimal: ControlSolution


def _square_matrix(values: np.ndarray, name: str) -> np.ndarray:
    """Copy a finite square array or raise an error naming the input."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _positive_horizon(value: int) -> int:
    """Accept an integer-valued positive horizon, excluding booleans/NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("T must be a positive integer") from error
    if (isinstance(value, (bool, np.bool_)) or not np.isfinite(number)
            or number < 1 or not number.is_integer()):
        raise ValueError("T must be a positive integer")
    return int(number)


def _positive_number(value: float, name: str) -> float:
    """Convert a scalar model parameter to a finite positive float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _state_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    """Read a flat/column state vector with one finite value per node."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != size or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def topology_metrics(adjacency: np.ndarray) -> dict[str, float | int]:
    """Compute scalar summaries of an undirected adjacency matrix with bct.

    Nonzero edges define binary density, degree, and clustering; strength
    uses absolute weights, preserving the tutorial's convention for signed
    uploads. Self-loops are ignored. Empty and one-node graphs are supported.
    """
    matrix = _square_matrix(adjacency, "A")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("A must be symmetric for undirected topology metrics")
    np.fill_diagonal(matrix, 0)
    size = len(matrix)
    binary = (matrix != 0).astype(float)
    density, _, edges = bct.density_und(binary) if size > 1 else (0.0, size, 0)
    return {
        "nodes": size,
        "edges": int(edges),
        "density": float(density),
        "mean_degree": float(np.mean(bct.degrees_und(binary))) if size else 0.0,
        "mean_strength": float(np.mean(bct.strengths_und(np.abs(matrix)))) if size else 0.0,
        "mean_clustering": float(np.mean(bct.clustering_coef_bu(binary))) if size else 0.0,
    }


def aggregate_control_energy(control_input: np.ndarray) -> np.ndarray:
    """Sum squared discrete control inputs over time for each node.

    The input has shape (time steps, nodes); the result has one value per
    node. This is a discrete sum, not continuous-time numerical integration.
    """
    values = np.asarray(control_input, dtype=float)
    if values.ndim != 2:
        raise ValueError("Control input must be a two-dimensional array")
    return np.square(values).sum(axis=0)


def state_trajectory(
    adjacency: np.ndarray, initial_state: np.ndarray, time_horizon: int
) -> np.ndarray:
    """Simulate uncontrolled discrete dynamics x(t+1) = A x(t).

    A must already be normalized. Return time_horizon rows, starting with
    the initial state at t=0; callers requesting states through t=T pass
    T+1. This also supports a single time point and performs no plotting.
    """
    matrix = _square_matrix(adjacency, "A")
    count = _positive_horizon(time_horizon)
    state = _state_vector(initial_state, len(matrix), "x0")
    trajectory = np.zeros((count, len(matrix)))
    for time_point in range(count):
        trajectory[time_point] = state
        state = matrix @ state
    return trajectory


def _solution(states: np.ndarray, inputs: np.ndarray, errors) -> ControlSolution:
    """Package a solver result and compute its per-node discrete energy."""
    return ControlSolution(
        states=states,
        inputs=inputs,
        energy=aggregate_control_energy(inputs),
        errors=(float(errors[0]), float(errors[1])),
    )


def compute_control_results(
    adjacency: np.ndarray,
    control: np.ndarray,
    constraint: np.ndarray,
    initial_state: np.ndarray,
    target_state: np.ndarray,
    time_horizon: int,
    *,
    c: float = 1.0,
    rho: float = 1.0,
) -> ControlResults:
    """Compute uncontrolled, minimum-energy, and optimal-control trajectories.

    A, B (control), and S (constraint) are N-by-N arrays in the same node
    order as the two state vectors. c controls nctpy's discrete normalization;
    T is a positive number of steps. Minimum energy uses S=0. Optimal control
    uses the supplied S and rho to penalize state magnitude and input cost.

    nctpy solves horizons of two or more steps. For T=1, solve B u = xf-A x0
    directly: fixed endpoints leave no intermediate state cost, so minimum
    and optimal inputs coincide. An unreachable one-step target raises
    ValueError. Solver residuals are retained in each ControlSolution.
    """
    matrix = _square_matrix(adjacency, "A")
    size = len(matrix)
    if not size:
        raise ValueError("The network has no nodes")
    B = _square_matrix(control, "B")
    S = _square_matrix(constraint, "S")
    if B.shape != matrix.shape or S.shape != matrix.shape:
        raise ValueError("B and S must have the same shape as A")
    x0 = _state_vector(initial_state, size, "x0")
    xf = _state_vector(target_state, size, "xf")
    horizon = _positive_horizon(time_horizon)
    normalization = _positive_number(c, "c")
    weight = _positive_number(rho, "rho")
    normalized = matrix_normalization(matrix, system="discrete", c=normalization)

    if horizon == 1:
        required_change = xf - normalized @ x0
        inputs = np.linalg.lstsq(B, required_change, rcond=None)[0]
        residual = float(np.linalg.norm(B @ inputs - required_change))
        if not np.allclose(B @ inputs, required_change, rtol=1e-7, atol=1e-9):
            raise ValueError("The target state is unreachable in one step with these controllers")
        minimum = _solution(np.vstack((x0, xf)), inputs.reshape(1, -1), (residual, residual))
        optimal = minimum
    else:
        arguments = dict(A_norm=normalized, T=horizon, B=B, x0=x0, xf=xf, system="discrete")
        minimum = _solution(*get_control_inputs(**arguments, rho=1.0, S=np.zeros_like(S)))
        optimal = _solution(*get_control_inputs(**arguments, rho=weight, S=S))

    return ControlResults(
        normalized_adjacency=normalized,
        uncontrolled=state_trajectory(normalized, x0, horizon + 1),
        minimum=minimum,
        optimal=optimal,
    )
