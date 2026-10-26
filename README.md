# Network Control Theory tutorial

An interactive Dash application for learning [Network Control Theory](https://github.com/BassettLab/control_package).

![Application preview](preview.png)

## Features

- Add and remove nodes without renumbering the nodes that remain.
- Connect selected nodes, remove selected edges, and edit edge weights.
- Select controller and constrained nodes.
- Edit or swap initial and target states.
- Upload a complete `A`, `B`, `S`, `c`, `T`, `x0`, `xf`, and `rho` configuration.
- Inspect density, degree, strength, and clustering as the topology changes.
- Plot uncontrolled trajectories, minimum/optimal control inputs, and aggregate energy per node.

## Run locally

```bash
conda env create -f environment.yml
conda activate nct-tutorial
python app.py
```

Open <http://localhost:8050>.

## Run with Docker

```bash
docker build -t nct-tutorial .
docker run --rm -p 8050:8050 nct-tutorial
```

Open <http://localhost:8050>. The container uses Miniconda and serves the app with Gunicorn.

## Upload format

Use a UTF-8 JSON file. `A` is required. `B` and `S` are optional diagonal matrices and default to identity matrices. `x0` and `xf` are optional vectors and default to zero. `c`, `T`, and `rho` are optional and default to `1`, `3`, and `1` respectively.

```json
{
  "A": [[0, 1], [1, 0]],
  "B": [[1, 0], [0, 1]],
  "S": [[1, 0], [0, 1]],
  "c": 1,
  "T": 3,
  "x0": [1, 0],
  "xf": [0, 1],
  "rho": 1
}
```

See [`example-config.json`](example-config.json) for a ready-to-upload example.

## Test

The core network and numerical helpers use the standard-library test runner:

```bash
python -m unittest discover -s tests -v
```
