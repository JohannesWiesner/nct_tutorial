import unittest

import numpy as np

from nct_utils import (
    add_node,
    aggregate_control_energy,
    delete_nodes,
    from_adjacency_matrix,
    matrices_from_elements,
    split_elements,
    state_trajectory,
    toggle_edges,
    topology_metrics,
    validate_configuration,
)


class ElementTests(unittest.TestCase):
    def setUp(self):
        self.adjacency = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]], dtype=float)
        self.elements = from_adjacency_matrix(self.adjacency)

    def test_elements_are_grouped_and_round_trip(self):
        self.assertEqual(set(self.elements), {"nodes", "edges"})
        adjacency, control, constraint, ids = matrices_from_elements(self.elements)
        np.testing.assert_array_equal(adjacency, self.adjacency)
        np.testing.assert_array_equal(control, np.eye(3))
        np.testing.assert_array_equal(constraint, np.eye(3))
        self.assertEqual(ids, ["0", "1", "2"])

    def test_flat_legacy_elements_are_partitioned_once(self):
        flat = self.elements["nodes"] + self.elements["edges"]
        grouped = split_elements(flat)
        self.assertEqual(len(grouped["nodes"]), 3)
        self.assertEqual(len(grouped["edges"]), 2)

    def test_deleting_node_preserves_other_ids_and_removes_edges(self):
        result = delete_nodes(self.elements, ["1"])
        self.assertEqual([node["data"]["id"] for node in result["nodes"]], ["0", "2"])
        self.assertEqual(result["edges"], [])

    def test_add_node_reuses_first_available_numeric_id(self):
        result = delete_nodes(self.elements, ["1"])
        result, new_id = add_node(result)
        self.assertEqual(new_id, "1")
        self.assertEqual(len(result["nodes"]), 3)

    def test_toggle_edges_adds_and_deletes_undirected_edge(self):
        selected_nodes = [{"id": "0"}, {"id": "2"}]
        result = toggle_edges(self.elements, selected_nodes, [], 3)
        self.assertEqual(len(result["edges"]), 3)
        added = next(edge for edge in result["edges"] if edge["data"]["id"] == "0--2")
        result = toggle_edges(result, [], [added["data"]], 3)
        self.assertEqual(len(result["edges"]), 2)

    def test_topology_metrics(self):
        metrics = topology_metrics(self.elements)
        self.assertEqual(metrics["nodes"], 3)
        self.assertEqual(metrics["edges"], 2)
        self.assertAlmostEqual(metrics["density"], 2 / 3)
        self.assertAlmostEqual(metrics["mean_degree"], 4 / 3)
        self.assertAlmostEqual(metrics["mean_clustering"], 0)


class ConfigurationTests(unittest.TestCase):
    def test_uploaded_configuration(self):
        result = validate_configuration(
            {
                "A": [[0, 1], [1, 0]],
                "B": [[1, 0], [0, 0]],
                "S": [[0, 0], [0, 1]],
                "x0": [1, 0],
                "xf": [0, 1],
                "c": 2,
                "T": 4,
                "rho": 3,
            }
        )
        _, control, constraint, _ = matrices_from_elements(result["elements"])
        np.testing.assert_array_equal(control, np.diag([1, 0]))
        np.testing.assert_array_equal(constraint, np.diag([0, 1]))
        self.assertEqual(result["states"][1]["xf"], 1)

    def test_rejects_non_symmetric_adjacency(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_configuration({"A": [[0, 1], [0, 0]]})


class NumericalTests(unittest.TestCase):
    def test_state_trajectory_supports_one_time_point(self):
        result = state_trajectory(np.eye(2), np.array([1, 2]), 1)
        np.testing.assert_array_equal(result, [[1, 2]])

    def test_state_trajectory_rejects_non_positive_horizon(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            state_trajectory(np.eye(2), np.array([1, 2]), 0)

    def test_aggregate_energy_sums_squared_input_over_time(self):
        result = aggregate_control_energy(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(result, [10, 20])


if __name__ == "__main__":
    unittest.main()
