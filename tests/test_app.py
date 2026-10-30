import base64
import copy
import json
import subprocess
import sys
import unittest

import numpy as np
from dash import no_update

from src.app import app, initial, render_network, update_figures
from src.graph_utils import validate_configuration
from src.nct_utils import compute_control_results
from src.view_utils import control_figure, energy_figure


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
            [sys.executable, "-c", "import sys, src.view_utils; "
             "assert not any(name in sys.modules for name in ('nctpy', 'bct', 'src.nct_utils'))"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class AppTests(unittest.TestCase):
    def callback_response(self, output, values, changed):
        """Exercise a registered Dash callback through its HTTP endpoint."""
        key = next(key for key in app.callback_map if output in key)
        callback = app.callback_map[key]
        response = app.server.test_client().post(
            "/_dash-update-component",
            json={
                "output": key,
                "outputs": [{"id": item.component_id, "property": item.component_property}
                            for item in callback["output"]],
                "inputs": [{**item, "value": values.get(f'{item["id"]}.{item["property"]}')}
                           for item in callback["inputs"]],
                "state": [{**item, "value": values.get(f'{item["id"]}.{item["property"]}')}
                          for item in callback["state"]],
                "changedPropIds": [changed],
            },
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()["response"]

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

    def test_default_plot_uses_continuous_time_for_both_control_methods(self):
        nodes = list(components(app.server.test_client().get("/_dash-layout").get_json()))
        props = {node["props"]["id"]: node["props"]
                 for node in nodes if "id" in node["props"]}
        self.assertEqual(initial["system"], "continuous")
        self.assertEqual(props["time-system"]["value"], "continuous")
        self.assertEqual(props["T-label"]["children"], "Time duration (T)")
        self.assertEqual((props["T"]["min"], props["T"]["step"]), (0.001, 0.001))
        result = update_figures(
            1, props["network-store"]["data"], props["states-table"]["data"],
            props["c"]["value"], props["T"]["value"], props["rho"]["value"],
            props["time-system"]["value"],
        )
        self.assertEqual(result[-1], "")
        self.assertTrue(all(figure.data for figure in result[:4]))
        for figure in result[:3]:
            self.assertEqual(figure.layout.xaxis.title.text, "Time (model units)")
            for trace in figure.data:
                self.assertEqual(len(trace.y), round(props["T"]["value"] / 0.001) + 1)
                self.assertAlmostEqual(trace.x[-1], props["T"]["value"])

    def test_uploaded_configuration_flows_through_model_and_view(self):
        store = validate_configuration({"A": [[0, 2], [2, 0]], "x0": [1, 0],
                                        "xf": [0, 1], "c": 3, "T": 4, "rho": 2})
        rendered = render_network(store, initial["elements"])
        self.assertEqual(rendered[3:6], (3, 4, 2))
        result = update_figures(1, store, store["states"], store["c"], store["T"], store["rho"], store["system"])
        self.assertEqual(result[-1], "")
        self.assertEqual(len(result[0].data[0].y), 5)
        self.assertEqual(len(result[3].data), 2)

    def test_bad_state_input_returns_feedback_instead_of_figures(self):
        rows = copy.deepcopy(initial["states"])
        rows[0]["x0"] = float("nan")
        result = update_figures(1, initial, rows, 1, 3, 1, "discrete")
        self.assertTrue(all(value is no_update for value in result[:4]))
        self.assertIn("x0", result[-1])

    def test_time_selector_updates_horizon_units_and_bounds(self):
        for system, label, step in (("continuous", "Time duration (T)", 0.001),
                                    ("discrete", "Time steps (T)", 1)):
            with self.subTest(system=system):
                response = self.callback_response(
                    "T-label.children", {"time-system.value": system}, "time-system.value"
                )
                self.assertEqual(response["T-label"]["children"], label)
                self.assertEqual(response["T"], {"min": step, "step": step})

    def test_continuous_upload_and_network_edit_preserve_selected_system(self):
        payload = {"A": [[0, 1], [1, 0]], "system": "continuous", "T": 1.25,
                   "x0": [1, 0], "xf": [0, 1]}
        upload = "data:application/json;base64," + base64.b64encode(json.dumps(payload).encode()).decode()
        loaded = self.callback_response(
            "network-store.data", {"upload-data.contents": upload}, "upload-data.contents"
        )["network-store"]["data"]
        rendered = self.callback_response(
            "cytoscape.elements",
            {"network-store.data": loaded, "cytoscape.elements": initial["elements"]},
            "network-store.data",
        )
        self.assertEqual(rendered["time-system"]["value"], "continuous")
        self.assertEqual(rendered["T"]["value"], 1.25)
        # Choose a new time system in the UI, then edit a graph whose store
        # still has the old system. The current selection must survive.
        edited = self.callback_response(
            "network-store.data",
            {"add-node.n_clicks": 1, "network-store.data": loaded,
             "states-table.data": loaded["states"], "c.value": 1, "T.value": 3,
             "rho.value": 1, "time-system.value": "discrete"},
            "add-node.n_clicks",
        )["network-store"]["data"]
        self.assertEqual((edited["system"], edited["T"]), ("discrete", 3))
        self.assertEqual(len(edited["elements"]["nodes"]), 3)
        self.assertEqual(render_network(edited, loaded["elements"])[-1], "discrete")

    def test_continuous_plots_use_model_times_for_all_trajectories(self):
        result = update_figures(1, initial, initial["states"], 1, 0.5, 1, "continuous")
        self.assertEqual(result[-1], "")
        for figure in result[:3]:
            self.assertEqual(figure.layout.xaxis.title.text, "Time (model units)")
            for trace in figure.data:
                self.assertEqual(len(trace.x), 501)
                self.assertAlmostEqual(trace.x[0], 0)
                self.assertAlmostEqual(trace.x[-1], 0.5)

    def test_plot_callback_uses_current_selection_without_a_graph_edit(self):
        response = self.callback_response(
            "state-trajectory.figure",
            {"plot.n_clicks": 1, "network-store.data": initial,
             "states-table.data": initial["states"], "c.value": 1,
             "T.value": 0.1, "rho.value": 1, "time-system.value": "continuous"},
            "plot.n_clicks",
        )
        self.assertEqual(response["plot-status"]["children"], "")
        for name in ("state-trajectory", "minimum-energy", "optimal-energy"):
            self.assertEqual(response[name]["figure"]["layout"]["xaxis"]["title"]["text"],
                             "Time (model units)")

    def test_switching_systems_updates_both_method_figures_without_mixing(self):
        for system, horizon in (("discrete", 3), ("continuous", 0.1), ("discrete", 4)):
            response = self.callback_response(
                "state-trajectory.figure",
                {"plot.n_clicks": 1, "network-store.data": initial,
                 "states-table.data": initial["states"], "c.value": 1,
                 "T.value": horizon, "rho.value": 1, "time-system.value": system},
                "plot.n_clicks",
            )
            for name, method in (("minimum-energy", "Minimum"), ("optimal-energy", "Optimal")):
                with self.subTest(system=system, method=method):
                    figure = response[name]["figure"]
                    self.assertEqual(figure["layout"]["title"]["text"],
                                     f"{method} control energy trajectories")
                    titled_axes = [axis for key, axis in figure["layout"].items()
                                   if key.startswith("xaxis") and axis.get("title", {}).get("text")]
                    self.assertEqual(len(titled_axes), 1)
                    self.assertEqual(titled_axes[0]["title"]["text"],
                                     "Time step" if system == "discrete" else "Time (model units)")
                    # The title belongs to the bottom panel, not the state panel above it.
                    anchor = titled_axes[0]["anchor"].replace("y", "yaxis", 1)
                    self.assertEqual(figure["layout"][anchor]["domain"][0], 0)
                    self.assertEqual(len(figure["data"]), 2 * len(initial["states"]))
                    self.assertEqual({trace["name"] for trace in figure["data"]},
                                     {row["id"] for row in initial["states"]})

    def test_minimum_figure_is_left_and_optimal_figure_is_right(self):
        nodes = list(components(app.server.test_client().get("/_dash-layout").get_json()))
        graphs = [node["props"] for node in nodes if node["type"] == "Graph"]
        self.assertEqual([graph["id"] for graph in graphs],
                         ["state-trajectory", "aggregate-energy", "minimum-energy", "optimal-energy"])
        self.assertEqual(graphs[2]["figure"]["layout"]["title"]["text"],
                         "Minimum control energy trajectories")
        self.assertEqual(graphs[3]["figure"]["layout"]["title"]["text"],
                         "Optimal control energy trajectories")

    def test_each_control_figure_contains_only_its_own_solution(self):
        for system, horizon in (("discrete", 3), ("continuous", 0.5)):
            store = validate_configuration({
                "A": [[0, 1], [1, 0]], "x0": [1, 0], "xf": [0, 1],
                "system": system, "T": horizon, "rho": 0.1,
            })
            model = compute_control_results(
                np.array([[0, 1], [1, 0]]), np.eye(2), np.eye(2), [1, 0], [0, 1],
                horizon, rho=0.1, system=system,
            )
            self.assertFalse(np.allclose(model.minimum.inputs, model.optimal.inputs))
            figures = update_figures(1, store, store["states"], 1, horizon, 0.1, system)
            for figure, solution in ((figures[1], model.minimum), (figures[2], model.optimal)):
                with self.subTest(system=system, title=figure.layout.title.text):
                    self.assertEqual(len(figure.data), 4)
                    for trace in figure.data:
                        node = int(trace.name)
                        is_state = "Type=State x" in trace.hovertemplate
                        values = solution.states if is_state else solution.inputs
                        times = model.state_times if is_state else model.input_times
                        np.testing.assert_allclose(trace.y, values[:, node])
                        np.testing.assert_allclose(trace.x, times)

    def test_swapping_edited_states_does_not_send_graph_or_style_updates(self):
        rows = copy.deepcopy(initial["states"])
        rows[0].update(x0=2.5, xf=-1.0)
        current_elements = copy.deepcopy(initial["elements"])
        current_elements["nodes"][0]["position"] = {"x": 100, "y": 200}
        current_elements["nodes"][0]["selected"] = True
        switched = self.callback_response(
            "network-store.data",
            {"switch-states.n_clicks": 1, "network-store.data": initial,
             "states-table.data": rows, "c.value": 2, "T.value": 3,
             "rho.value": 4, "time-system.value": "discrete"},
            "switch-states.n_clicks",
        )["network-store"]["data"]
        self.assertEqual(switched["states"][0], {"id": "0", "x0": -1.0, "xf": 2.5})
        rendered = self.callback_response(
            "cytoscape.elements",
            {"network-store.data": switched, "cytoscape.elements": current_elements},
            "network-store.data",
        )
        self.assertNotIn("cytoscape", rendered)
        self.assertNotIn("topology-metrics", rendered)
        self.assertEqual(rendered["states-table"]["data"], switched["states"])
        self.assertEqual(rendered["c"]["value"], 2)

    def test_graph_data_changes_still_refresh_elements_styles_and_metrics(self):
        for change in ("weight", "controller", "constrain"):
            modified = copy.deepcopy(initial)
            if change == "weight":
                modified["elements"]["edges"][0]["data"]["weight"] = 4.5
            else:
                modified["elements"]["nodes"][0]["data"][change] = False
            with self.subTest(change=change):
                rendered = self.callback_response(
                    "cytoscape.elements",
                    {"network-store.data": modified, "cytoscape.elements": initial["elements"]},
                    "network-store.data",
                )
                self.assertEqual(rendered["cytoscape"]["elements"], modified["elements"])
                self.assertIn("stylesheet", rendered["cytoscape"])
                self.assertIn("topology-metrics", rendered)

    def test_fractional_duration_is_rejected_after_switch_to_discrete(self):
        result = update_figures(1, initial, initial["states"], 1, 0.5, 1, "discrete")
        self.assertTrue(all(value is no_update for value in result[:4]))
        self.assertIn("positive integer for discrete time", result[-1])

    def test_viewport_asset_is_served_from_src_application(self):
        client = app.server.test_client()
        self.assertIn(b"/assets/viewport.js", client.get("/").data)
        response = client.get("/assets/viewport.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"keepVisible", response.data)
        response.close()


if __name__ == "__main__":
    unittest.main()
