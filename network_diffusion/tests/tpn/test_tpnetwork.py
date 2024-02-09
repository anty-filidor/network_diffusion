"""Tests for the network_diffusion.tpn.tpnetwork."""

import unittest

import numpy as np

from network_diffusion.tpn import TemporalNetwork


class TestTemporalNetwork(unittest.TestCase):
    """Test TemporalNetwork class."""

    def setUp(self) -> None:
        self.net = TemporalNetwork.from_cogsnet(
            "exponential",
            5,
            70,
            0.3,
            0.1,
            3600,
            "network_diffusion/tests/data/cogsnet_data.csv",
            ";",
        )

    def test_from_cogsnet_number_of_snapshots(self):
        self.assertEqual(len(self.net.snaps), 8)

    def test_from_cogsnet_number_of_edges(self):
        edges = [len(snap.layers["layer_1"].edges) for snap in self.net.snaps]
        self.assertEqual(edges, [9] * 8)

    def test_from_cogsnet_number_of_nodes(self):
        nodes = [len(snap.layers["layer_1"].nodes) for snap in self.net.snaps]
        self.assertEqual(nodes, [3] * 8)

    def test_from_cogsnet_mean_weight(self):
        exp_mean_weights = np.array(
            [
                0.11333333121405707,
                0.11333333121405707,
                0.23553334342108834,
                0.31820668114556205,
                0.31820668114556205,
                0.31820668114556205,
                0.3294113477071126,
                0.3372546169492934,
            ]
        )
        mean_weights = []
        for snap in self.net.snaps:
            l_graph = snap.layers["layer_1"]
            weights = [e[-1]["weight"] for e in l_graph.edges.data()]
            mean_weights.append(np.mean(weights))
        np.testing.assert_almost_equal(
            np.array(mean_weights), exp_mean_weights, decimal=3
        )
