# Network Control Theory tutorial

An interactive Dash application for learning [Network Control Theory](https://github.com/LindenParkesLab/nctpy).

![Application preview](preview.png)

## Features

- Add and remove nodes without renumbering the nodes that remain.
- Pan and zoom with automatic recovery if the graph leaves the canvas; use **Fit graph** to show all nodes without changing their positions.
- Connect selected nodes, remove selected edges, and edit edge weights.
- Select controller and constrained nodes.
- Edit or swap initial and target states while preserving the graph layout.
- Start in continuous time, or choose discrete time, with matching dynamics and energy calculations.
- Upload a complete `A`, `B`, `S`, `c`, `T`, `x0`, `xf`, `rho`, and `system` configuration.
- Inspect density, degree, strength, and clustering as the topology changes.
- Compare minimum-energy trajectories in the lower-left plot and optimal-control trajectories in the lower-right plot for the selected time system.
- Inspect uncontrolled trajectories and aggregate energy per node for the selected time system under **Control Energy Measures**.

## Run locally

```bash
conda env create -f environment.yml
conda activate nct-tutorial
python -m src.app
```

Run these commands from the repository root, then open <http://localhost:8050>.
For Gunicorn, use `gunicorn --bind 0.0.0.0:8050 src.app:server`.

## Run with Docker

```bash
docker build -t nct-tutorial .
docker run --rm -p 8050:8050 nct-tutorial
```

Open <http://localhost:8050>. The container uses Miniconda and serves the app with Gunicorn.

## Upload format

Use a UTF-8 JSON file. `A` is required. `B` and `S` are optional diagonal matrices and default to identity matrices. `x0` and `xf` are optional vectors and default to zero. `c`, `T`, and `rho` are optional and default to `1`, `3`, and `1` respectively.

`system` is optional and accepts `"discrete"` or `"continuous"`.
Discrete `T` must be a positive integer; continuous `T` is a positive duration
in increments of `0.001`, for example `0.5` or `1.25`. Older JSON files without
`system` continue to use discrete time for compatibility, even though new app
sessions start in continuous time. Uploading a configuration also updates
the time-system selector and the horizon label.

```json
{
  "A": [[0, 1], [1, 0]],
  "B": [[1, 0], [0, 1]],
  "S": [[1, 0], [0, 1]],
  "c": 1,
  "T": 3,
  "x0": [1, 0],
  "xf": [0, 1],
  "rho": 1,
  "system": "discrete"
}
```

See [`example-config.json`](example-config.json) for a ready-to-upload example.

## Code organization

| Module | Responsibility |
| --- | --- |
| `src/nct_utils.py` | Numerical model: nctpy normalization/control solutions, bct topology metrics, and energy aggregation. Accepts arrays and returns numerical results. |
| `src/graph_utils.py` | Cytoscape editing, upload validation, UI state, and conversion between graph elements and model arrays. |
| `src/view_utils.py` | Cytoscape styles, persistent parameter labels, topology cards, and Plotly figures. Consumes precomputed metrics, energies, and time coordinates. |
| `src/time_utils.py` | Shared time-system validation and sampling conventions. |
| `src/app.py` | Dash layout and controller callbacks that coordinate the helpers. |

Application modules are grouped in the `src` package; tests remain under `tests/`.
Import helpers using paths such as `from src.graph_utils import node_ids`.
The app resolves `assets/` relative to the repository root, including in Docker.
Each helper documents its input format, return values, and relevant conventions.

## Numerical conventions

The **Time system** selector applies to normalization, uncontrolled trajectories,
both control solutions, time axes, and energy calculations. New app sessions
default to continuous time. Short descriptions beneath each parameter explain
its role. After changing a parameter or system, click **Plot trajectories and
energy** to recompute the figures.

The lower-left figure is reserved for **minimum control energy** trajectories;
the lower-right figure is reserved for **optimal control energy** trajectories.
Each contains only its own method's states and inputs. Both use the selected
time system and update when you click **Plot trajectories and energy**. The
method appears in each title, with one time-axis title beneath the bottom
panel indicating the time units. Minimum and optimal control are available in
both time systems.

| Convention | Discrete | Continuous |
| --- | --- | --- |
| Dynamics | `x(t+1) = A_normalized x(t) + B u(t)` | `dx/dt = A_normalized x(t) + B u(t)` |
| Horizon `T` | Positive integer number of transitions | Positive duration in model time units, in increments of `0.001` |
| State/input samples | `T+1` states and `T` inputs | Both sampled every `0.001`, including `t=0` and `t=T` |
| Per-node energy | Sum of squared inputs | Integral of squared inputs over model time |

`c` is passed explicitly to nctpy's normalization for the selected system.
Continuous normalization includes the identity subtraction required for stable
continuous dynamics. The uncontrolled continuous trajectory uses a matrix
exponential at each sample interval. Continuous energy uses nctpy's Simpson
integration multiplied by `0.001`; its unscaled `integrate_u` result integrates
over sample indices rather than model time. See the nctpy documentation on
[time systems](https://control-package.readthedocs.io/en/latest/guide/time_systems.html) and
[energy measures](https://control-package.readthedocs.io/en/latest/guide/energies.html).

`rho` weights control-input cost in the optimal solution; minimum-energy control
uses a zero state-constraint matrix. For discrete `T=1`, the tutorial solves
`B u = xf - A_normalized x0` directly and reports unreachable targets as input
errors. Continuous `T=1` uses the continuous solver. Both control solutions
retain the solver's numerical residuals.

Topology cards use bct's undirected density, degree, and binary clustering.
Mean strength uses absolute edge weights. Self-loops are excluded from these
summaries, and empty/single-node networks return zero connectivity measures.

## Test

After installing the app requirements, run the model, graph-adapter, view,
and callback tests with the standard-library test runner:

```bash
python -m unittest discover -s tests -v
```

Viewport logic tests use the real Cytoscape.js engine in headless mode (Node.js required):

```bash
npm install --no-save --package-lock=false cytoscape@3.33.1
node --test tests/test_viewport.cjs
```

Browser regression tests cover viewport recovery, zoom limits, resizing, uploads,
the **Fit graph** button, and preservation of node positions when swapping states.
Install the app requirements first, then run:

```bash
python -m pip install playwright
python -m playwright install chromium
python -m unittest discover -s tests/browser -v
```
