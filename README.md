# Network Control Theory tutorial

An interactive Dash application for learning [Network Control Theory](https://github.com/LindenParkesLab/nctpy).

![Application preview](preview.png)

## Features

- Add and remove nodes from the adjacency matrix
- Connect selected nodes, remove selected edges, and edit edge weights.
- Select controller and state-penalty nodes. Blue lightning badges on the upper-left rim indicate control input (B); amber targets on the upper-right rim indicate state penalty (S). Node numbers stay centered, and a separate purple outline indicates selection.
- Edit or swap initial and target states while preserving the graph layout.
- Use **Arrange graph** to reorganize scattered or overlapping nodes, or **Fit graph** to adjust only the view.
- Start in continuous time with duration T = 1, or choose discrete time, with matching dynamics and energy calculations.
- Read node IDs without edge-label clutter; edge widths still encode their weights.
- Edit states, parameters, and the network in controls that wrap to fit the screen.
- Upload a complete `A`, `B`, `S`, `c`, `T`, `x0`, `xf`, `rho`, and `system` configuration.
- Inspect density, degree, strength, and clustering as the topology changes.
- Compare minimum-energy trajectories in the lower-left plot and optimal-control trajectories in the lower-right plot for the selected time system.
- Inspect uncontrolled trajectories and aggregate energy per node for the selected time system under **Control Energy Measures**.

## Graph arrangement

**Arrange graph** runs a bounded fCoSE force-directed layout, clears remaining
overlaps between node/badge bounds, and fits the result. The layout stops once
finished. Nodes remain draggable; dragging, clicking the canvas, panning, or
zooming during an animation cancels it without a delayed fit. Resizing lets the
arrangement finish and fit the new canvas size. **Fit graph** cancels an active
arrangement and changes only pan and zoom.

One client-side controller handles initial layout, manual arrangement, and
batched node/edge additions or removals. State edits, state swaps, parameter
changes, role toggles, edge-weight edits, and plotting do not restart it. B/S
roles, selection, graph data, and numerical calculations retain their meaning;
S is a state penalty, not a visual pin.

The extra-layout bundle supplied by Dash Cytoscape provides fCoSE. Native
`autoRefreshLayout` is disabled, and automatic fitting is deferred while the
controller is busy. Finished positions are synchronized to Dash; server updates
omit coordinates for existing nodes so stale callbacks cannot undo a layout or
drag.

The optional comparison below exercises CoSE, fCoSE, and the complete arrangement
on the tutorial network, overlapping/spread nodes, disconnected components,
isolated nodes, and a dense 150-node graph. It reports runtime, occupied bounds,
and badge overlaps. These headless measurements are not browser performance
benchmarks; the final spacing pass addresses overlaps left by the solver.

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

Use a UTF-8 JSON file. `A` is required. `B` and `S` are optional diagonal matrices and default to identity matrices. `x0` and `xf` are optional vectors and default to zero. `c` and `rho` default to `1`. If omitted, `T` defaults to `1` for continuous time and `3` for discrete time; an explicitly supplied `T` is preserved.

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

## Checks

Run the Python model, view, and callback regressions from the repository root:

```bash
python -m unittest discover -s tests -q
```

The optional JavaScript checks use the Cytoscape and React wrapper versions
bundled with Dash Cytoscape 1.0.2. They verify badge styles, role updates,
selection, viewport fitting, cancellation, and arrangement with the real graph
implementation:

```bash
npm install --no-save --package-lock=false cytoscape@3.28.1 react-cytoscapejs@2.0.0 cytoscape-fcose@2.2.0
node --test tests/test_*.cjs
```

To reproduce the optional layout comparison:

```bash
node tests/benchmark_layouts.cjs
```

To check arrangement, interruption, uploads, dragging, role toggles, editable
controls, and responsive widths in a browser:

```bash
python -m pip install playwright
python -m playwright install chromium
python -m unittest discover -s tests/browser -v
```
