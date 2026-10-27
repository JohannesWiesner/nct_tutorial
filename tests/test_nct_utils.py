import subprocess
import sys
import unittest

import numpy as np

from nct_utils import (
    aggregate_control_energy, compute_control_results, state_trajectory, topology_metrics,
)


class TopologyTests(unittest.TestCase):
    def test_weighted_chain(self):
        metrics = topology_metrics(np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]]))
        self.assertEqual(metrics["nodes"], 3)
        self.assertEqual(metrics["edges"], 2)
        self.assertAlmostEqual(metrics["density"], 2 / 3)
        self.assertAlmostEqual(metrics["mean_degree"], 4 / 3)
        self.assertAlmostEqual(metrics["mean_strength"], 2)
        self.assertAlmostEqual(metrics["mean_clustering"], 0)

    def test_triangle_uses_binary_clustering_and_absolute_strength(self):
        adjacency = np.array([[0, -2, 3], [-2, 0, 1], [3, 1, 0]])
        before = adjacency.copy()
        metrics = topology_metrics(adjacency)
        self.assertEqual(metrics["density"], 1)
        self.assertEqual(metrics["mean_degree"], 2)
        self.assertEqual(metrics["mean_strength"], 4)
        self.assertEqual(metrics["mean_clustering"], 1)
        np.testing.assert_array_equal(adjacency, before)

    def test_empty_and_single_node_graphs(self):
        for size in (0, 1):
            with self.subTest(size=size):
                metrics = topology_metrics(np.zeros((size, size)))
                self.assertEqual(metrics["nodes"], size)
                self.assertTrue(all(value == 0 for key, value in metrics.items() if key != "nodes"))


class NumericalTests(unittest.TestCase):
    def test_state_trajectory_supports_one_time_point(self):
        np.testing.assert_array_equal(state_trajectory(np.eye(2), [1, 2], 1), [[1, 2]])

    def test_state_trajectory_rejects_non_positive_horizon(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            state_trajectory(np.eye(2), [1, 2], 0)

    def test_aggregate_energy_sums_squared_input_over_time(self):
        np.testing.assert_array_equal(aggregate_control_energy([[1, 2], [3, 4]]), [10, 20])


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.B = np.eye(2)
        self.S = np.eye(2)
        self.x0 = np.array([1.0, 0.0])
        self.xf = np.array([0.0, 1.0])

    def solve(self, **kwargs):
        values = dict(adjacency=self.A, control=self.B, constraint=self.S,
                      initial_state=self.x0, target_state=self.xf, time_horizon=3)
        values.update(kwargs)
        return compute_control_results(**values)

    def test_control_solutions_obey_discrete_dynamics_and_endpoints(self):
        result = self.solve()
        self.assertEqual(result.uncontrolled.shape, (4, 2))
        for solution in (result.minimum, result.optimal):
            np.testing.assert_allclose(solution.states[0], self.x0, atol=1e-10)
            np.testing.assert_allclose(solution.states[-1], self.xf, atol=1e-10)
            self.assertEqual(solution.inputs.shape, (3, 2))
            np.testing.assert_allclose(
                solution.states[1:],
                solution.states[:-1] @ result.normalized_adjacency.T + solution.inputs @ self.B.T,
                atol=1e-10,
            )
            self.assertTrue(all(error < 1e-8 for error in solution.errors))

    def test_zero_adjacency_has_known_minimum_input_and_energy(self):
        target = np.array([2.0, -3.0])
        result = self.solve(adjacency=np.zeros((2, 2)), target_state=target)
        np.testing.assert_allclose(result.minimum.inputs[:-1], 0, atol=1e-10)
        np.testing.assert_allclose(result.minimum.inputs[-1], target, atol=1e-10)
        np.testing.assert_allclose(result.minimum.energy, [4, 9], atol=1e-10)

    def test_normalization_parameter_changes_dynamics(self):
        first, second = self.solve(c=1), self.solve(c=3)
        np.testing.assert_allclose(first.normalized_adjacency, self.A / 2)
        np.testing.assert_allclose(second.normalized_adjacency, self.A / 4)
        self.assertFalse(np.allclose(first.uncontrolled, second.uncontrolled))

    def test_rho_affects_optimal_inputs_but_not_minimum_energy(self):
        low, high = self.solve(rho=0.1), self.solve(rho=10)
        np.testing.assert_allclose(low.minimum.inputs, high.minimum.inputs)
        self.assertFalse(np.allclose(low.optimal.inputs, high.optimal.inputs))

    def test_zero_constraints_reduce_optimal_to_minimum_energy(self):
        result = self.solve(constraint=np.zeros((2, 2)), rho=3)
        np.testing.assert_allclose(result.minimum.inputs, result.optimal.inputs, atol=1e-10)

    def test_one_step_solution_preserves_supported_horizon(self):
        result = self.solve(time_horizon=1)
        expected = self.xf - result.normalized_adjacency @ self.x0
        np.testing.assert_allclose(result.minimum.inputs, expected.reshape(1, -1))
        np.testing.assert_allclose(result.minimum.energy, expected ** 2)
        np.testing.assert_allclose(result.optimal.inputs, result.minimum.inputs)
        self.assertEqual(result.uncontrolled.shape, (2, 2))

    def test_one_step_rejects_an_unreachable_target(self):
        with self.assertRaisesRegex(ValueError, "unreachable"):
            self.solve(control=np.zeros((2, 2)), time_horizon=1)

    def test_partial_controller_set_reaches_target_over_multiple_steps(self):
        control = np.diag([1.0, 0.0])
        result = self.solve(control=control)
        for solution in (result.minimum, result.optimal):
            np.testing.assert_allclose(solution.states[-1], self.xf, atol=1e-8)
            np.testing.assert_allclose(solution.inputs[:, 1], 0, atol=1e-8)

    def test_invalid_parameters_raise_readable_errors(self):
        for key, value, message in (
            ("c", 0, "c"), ("c", np.nan, "c"),
            ("rho", -1, "rho"), ("rho", np.inf, "rho"),
            ("time_horizon", 1.5, "T"), ("time_horizon", True, "T"),
            ("initial_state", [np.nan, 0], "x0"),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                self.solve(**{key: value})

    def test_model_import_does_not_load_ui_libraries(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, nct_utils; "
             "assert not any(name in sys.modules for name in ('dash', 'plotly', 'dash_cytoscape'))"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
