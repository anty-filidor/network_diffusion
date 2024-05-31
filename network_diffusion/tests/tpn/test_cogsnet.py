"""Tests for the network_diffusion.tpn.cogsnet_lib."""

import unittest

from network_diffusion.tpn.cogsnet_lib import _cogsnet


class TestCogsnet(unittest.TestCase):
    """Test TemporalNetwork class."""

    def test_returned_number_of_snapshots(self):
        snapshot_interval = 0
        tn = _cogsnet(
            "exponential",
            snapshot_interval,
            72,
            0.3,
            0.1,
            3600,
            "network_diffusion/tests/data/cogsnet_data.csv",
            ";",
        )
        self.assertEqual(len(tn), 8)

        snapshot_interval = 10
        tn = _cogsnet(
            "exponential",
            snapshot_interval,
            72,
            0.3,
            0.1,
            3600,
            "network_diffusion/tests/data/cogsnet_data.csv",
            ";",
        )
        self.assertEqual(len(tn), 4)

    def test_returned_number_of_edges(self):
        number_of_nodes = 3
        number_of_edges = number_of_nodes**2
        tn = _cogsnet(
            "exponential",
            10,
            72,
            0.3,
            0.1,
            3600,
            "network_diffusion/tests/data/cogsnet_data.csv",
            ";",
        )
        for i in range(0, len(tn)):
            self.assertEqual(len(tn[i]), number_of_edges)
