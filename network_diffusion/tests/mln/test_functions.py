#!/usr/bin/env python3
# pylint: disable-all
# type: ignore

"""Tests for the network_diffusion.mln.functions."""

import os
import unittest

import networkx as nx

from network_diffusion import MultilayerNetwork, utils
from network_diffusion.mln.functions import multiplexing_coefficient


class TestFunctions(unittest.TestCase):
    """Test functions."""

    def setUp(self):
        """Set up most common testing parameters."""
        self.network = MultilayerNetwork.from_mpx(
            os.path.join(
                utils.get_absolute_path(), "tests/data/florentine.mpx"
            )
        )

    def test_compute_multiplexing_coefficient(self):
        """Tests if multiplexing coefficient is computed correctly."""
        names = ["a", "b"]
        network = MultilayerNetwork.from_nx_layer(
            nx.les_miserables_graph(), names
        )
        mcf = multiplexing_coefficient(net=network)
        self.assertEqual(
            mcf,
            1,
            "Multiplexing coefficient for multpilex network should be 1",
        )

        mcf = multiplexing_coefficient(self.network)
        self.assertAlmostEqual(
            mcf,
            0.733,
            2,
            'Multiplexing coefficient for "florentine" network should be '
            "almost 0.73",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
