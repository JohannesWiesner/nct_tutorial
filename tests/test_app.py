import copy
import subprocess
import sys
import unittest

import numpy as np
from dash import no_update

from app import app, initial, render_network, update_figures
from graph_utils import validate_configuration
from view_utils import control_figure, energy_figure


def components(item):
    """Walk the serialized Dash layout, including nested component children."""
    if isinstance(item, dict):
        if "props" in item:
            yield item
        for value in item.values():
            yield from components(value)
    elif isinstance(item, list):
        for value in item:
            yield from components(value)


class ViewTests(unittest.TestCase):
    def test_energy_chart_uses_precomputed_values_without_resquaring(self):
        figure = energy_figure(np.array([4, 9]), np.array([5, 10]), ["2", "10"])
        traces = {trace.name: trace for trace in figure.data}
        np.testing.assert_array_equal(traces["Minimum"].y, [4, 9])
        np.testing.assert_array_equal(traces["Optimal"].y, [5, 10])
        self.assertEqual(list(traces["Minimum"].x), ["2", "10"])

    def test_control_chart_supports_T_inputs_and_T_plus_one_states(self):
        figure = control_figure(np.ones((4, 2)), np.ones((3, 2)), ["0", "2"], "Control")
        self.assertEqual(sorted(len(trace.y) for trace in figure.data), [3, 3, 4, 4])

    def test_view_import_does_not_load_numerical_backends(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, view_utils; "
             "assert not any(name in sys.modules for name in ('nctpy', 'bct', 'nct_utils'))"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class AppTests(unittest.TestCase):
    def test_parameter_labels_are_associated_with_populated_inputs(self):
        response = app.server.test_client().get("/_dash-layout")
        self.assertEqual(response.status_code, 200)
        nodes = list(components(response.get_json()))
        labels = {node["props"].get("htmlFor"): node["props"]["children"]
                  for node in nodes if node["type"] == "Label"}
        inputs = {node["props"].get("id"): node["props"]
                  for node in nodes if node["type"] == "Input"}
        for name in ("c", "T", "rho"):
            with self.subTest(parameter=name):
                self.assertIn(name, labels)
                self.assertIn(name, labels[name])
                self.assertEqual(inputs[name]["value"], initial[name])
                self.assertNotIn("placeholder", inputs[name])

    def test_all_four_figures_render_for_default_network(self):
        result = update_figures(1, initial, initial["states"], 1, 3, 1)
        self.assertEqual(result[-1], "")
        self.assertTrue(all(figure.data for figure in result[:4]))
        self.assertEqual(len(result[0].data[0].y), 4)

    def test_uploaded_configuration_flows_through_model_and_view(self):
        store = validate_configuration({"A": [[0, 2], [2, 0]], "x0": [1, 0],
                                        "xf": [0, 1], "c": 3, "T": 4, "rho": 2})
        rendered = render_network(store)
        self.assertEqual(rendered[3:6], (3, 4, 2))
        result = update_figures(1, store, store["states"], store["c"], store["T"], store["rho"])
        self.assertEqual(result[-1], "")
        self.assertEqual(len(result[0].data[0].y), 5)
        self.assertEqual(len(result[3].data), 2)

    def test_bad_state_input_returns_feedback_instead_of_figures(self):
        rows = copy.deepcopy(initial["states"])
        rows[0]["x0"] = float("nan")
        result = update_figures(1, initial, rows, 1, 3, 1)
        self.assertTrue(all(value is no_update for value in result[:4]))
        self.assertIn("x0", result[-1])


if __name__ == "__main__":
    unittest.main()
