import subprocess
import sys
import unittest

import numpy as np
from scipy.linalg import expm

from src.nct_utils import (
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

    def test_continuous_energy_integrates_over_duration_not_sample_count(self):
        inputs = np.tile([2.0, 3.0], (501, 1))
        np.testing.assert_allclose(
            aggregate_control_energy(inputs, system="continuous"), [2.0, 4.5]
        )


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
        np.testing.assert_array_equal(result.state_times, [0, 1, 2, 3])
        np.testing.assert_array_equal(result.input_times, [0, 1, 2])
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

    def test_continuous_normalization_and_uncontrolled_trajectory(self):
        result = self.solve(system="continuous", time_horizon=0.25, c=3)
        expected_matrix = self.A / 4 - np.eye(2)
        np.testing.assert_allclose(result.normalized_adjacency, expected_matrix)
        self.assertEqual(result.system, "continuous")
        self.assertEqual(result.uncontrolled.shape, (251, 2))
        self.assertAlmostEqual(result.state_times[-1], 0.25)
        np.testing.assert_array_equal(result.state_times, result.input_times)
        for index in (0, 100, 250):
            expected = expm(expected_matrix * result.state_times[index]) @ self.x0
            np.testing.assert_allclose(result.uncontrolled[index], expected, atol=1e-12)

    def test_continuous_control_obeys_differential_dynamics_and_endpoints(self):
        result = self.solve(system="continuous", time_horizon=0.25)
        for solution in (result.minimum, result.optimal):
            np.testing.assert_allclose(solution.states[0], self.x0, atol=1e-10)
            np.testing.assert_allclose(solution.states[-1], self.xf, atol=1e-10)
            self.assertEqual(solution.inputs.shape, (251, 2))
            derivative = np.gradient(solution.states, result.state_times, axis=0, edge_order=2)
            expected = solution.states @ result.normalized_adjacency.T + solution.inputs @ self.B.T
            np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=1e-6)
            self.assertTrue(all(error < 1e-8 for error in solution.errors))

    def test_continuous_minimum_input_and_energy_match_analytic_solution(self):
        duration = 0.5
        target = np.array([2.0, -3.0])
        result = self.solve(
            adjacency=np.zeros((2, 2)), initial_state=np.zeros(2),
            target_state=target, time_horizon=duration, system="continuous",
        )
        # Normalization gives A=-I. Its finite-horizon Gramian is w*I.
        gramian = (1 - np.exp(-2 * duration)) / 2
        expected_inputs = np.exp(-(duration - result.input_times[:, None])) * target / gramian
        np.testing.assert_allclose(result.minimum.inputs, expected_inputs, atol=1e-10)
        np.testing.assert_allclose(result.minimum.energy, target ** 2 / gramian, rtol=1e-10)

    def test_continuous_T_one_uses_continuous_solver_and_honors_rho(self):
        low = self.solve(system="continuous", time_horizon=1, rho=0.1)
        high = self.solve(system="continuous", time_horizon=1, rho=10)
        self.assertEqual(low.minimum.inputs.shape, (1001, 2))
        np.testing.assert_allclose(low.minimum.inputs, high.minimum.inputs)
        self.assertFalse(np.allclose(low.optimal.inputs, high.optimal.inputs))

    def test_continuous_zero_constraints_reduce_optimal_to_minimum(self):
        result = self.solve(system="continuous", constraint=np.zeros((2, 2)), rho=3, time_horizon=0.1)
        np.testing.assert_allclose(result.minimum.inputs, result.optimal.inputs, atol=1e-10)

    def test_continuous_partial_controllers_reach_target(self):
        result = self.solve(system="continuous", control=np.diag([1.0, 0.0]), time_horizon=1)
        for solution in (result.minimum, result.optimal):
            np.testing.assert_allclose(solution.states[-1], self.xf, atol=1e-8)
            np.testing.assert_allclose(solution.inputs[:, 1], 0, atol=1e-8)

    def test_invalid_time_system_and_continuous_horizons_are_rejected(self):
        for system in (None, "invalid", "Continuous"):
            with self.subTest(system=system), self.assertRaisesRegex(ValueError, "Time system"):
                self.solve(system=system)
        for horizon in (0, -1, np.inf, np.nan, True, None, 0.0001, 0.1001):
            with self.subTest(T=horizon), self.assertRaisesRegex(ValueError, "duration"):
                self.solve(system="continuous", time_horizon=horizon)

    def test_shortest_continuous_duration_retains_both_endpoints(self):
        result = self.solve(system="continuous", time_horizon=0.001)
        np.testing.assert_allclose(result.state_times, [0, 0.001])
        np.testing.assert_allclose(result.minimum.states[-1], self.xf, atol=1e-10)

    def test_model_import_does_not_load_ui_libraries(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, src.nct_utils; "
             "assert not any(name in sys.modules for name in ('dash', 'plotly', 'dash_cytoscape'))"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
