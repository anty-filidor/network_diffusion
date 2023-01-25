#!/usr/bin/env python3
# pylint: disable-all
# type: ignore

"""Tests for the network_diffusion.mln.mlnetwork."""

import os
import unittest

import networkx as nx

from network_diffusion import MultilayerNetwork, utils
from network_diffusion.models.utils.compartmental import CompartmentalGraph


class TestMultilayerNetwork(unittest.TestCase):
    """Test MultilayerNetwork class."""

    def setUp(self):
        """Set up most common testing parameters."""
        self.network = MultilayerNetwork.from_mpx(
            os.path.join(
                utils.get_absolute_path(), "tests/data/florentine.mpx"
            )
        )

    def test_from_mpx(self):
        """Tests loading network from mpx file."""
        network = MultilayerNetwork.from_mpx(
            os.path.join(
                utils.get_absolute_path(), "tests/data/bankwiring.mpx"
            )
        )

        exp_layers = {
            "horseplay",
            "arguments",
            "friendship",
            "antagonist",
            "help",
            "job_trading",
        }
        real_layers = set(network.layers.keys())
        self.assertEqual(
            real_layers,
            exp_layers,
            f"Layers should be equal ({real_layers} !={exp_layers})",
        )

        self.assertEqual(
            network.layers["horseplay"].nodes["W1"]["status"],
            None,
            "Node should have None in status attr",
        )

    def test_from_nx_layers(self):
        """Tests loading network from several graphs."""
        graphs = [nx.random_tree(10)] * 4
        layer_names = ["1", "2", "3", "4"]

        network = MultilayerNetwork.from_nx_layers(graphs, layer_names)
        exp_res = set(network.layers.keys())

        self.assertEqual(
            set(layer_names),
            exp_res,
            f"Layers should be equal ({exp_res} !={layer_names})",
        )

        self.assertEqual(
            network.layers["1"].nodes[0]["status"],
            None,
            "Node should have None in status attr",
        )

    def test_from_nx_layer(self):
        """Checks if creating MLN is correct by from_nx_layer func."""
        names = ["a", "b"]
        network = MultilayerNetwork.from_nx_layer(
            nx.les_miserables_graph(), names
        )

        self.assertTrue(
            network.layers["a"].nodes == network.layers["b"].nodes,
            'Nodes of layer "a" and "b" should be equal!',
        )
        self.assertTrue(
            network.layers["a"].edges == network.layers["b"].edges,
            'Edges of layer "a" and "b" should be equal!',
        )
        self.assertFalse(
            network.layers["a"] == network.layers["b"],
            'Network in layer "a" should be a diffeereent object than in "b"!',
        )

        self.assertEqual(
            network.layers["a"].nodes["Napoleon"]["status"],
            None,
            "Node should have None in status attr",
        )

    def test_get_states_num(self):
        """Tests if nodes states are being returned correctly."""
        exp_result = {"marriage": ((None, 15),), "business": ((None, 11),)}
        self.assertEqual(
            self.network.get_states_num(),
            exp_result,
            "Network states are not same as expected.",
        )

    def test__actor_to_compartment_state(self):
        """Tests if node state is being returned correctly."""

        self.assertEqual(
            CompartmentalGraph()._actor_to_compartment_state(
                self.network.get_actor("Acciaiuoli")
            ),
            ("marriage.None",),
            "Nodes states are not same as expected.",
        )

    def test_get_layer_names(self):
        """Tests if layer names are read correctly."""
        self.assertEqual(
            self.network.get_layer_names(),
            ["marriage", "business"],
            "Incorrect layer names",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
