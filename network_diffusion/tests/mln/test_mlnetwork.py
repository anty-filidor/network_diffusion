#!/usr/bin/env python3
# pylint: disable-all
# type: ignore

"""Tests for the network_diffusion.mln.mlnetwork."""

import copy
import os
import unittest

import networkx as nx
import numpy as np

from network_diffusion import utils
from network_diffusion.mln import MLNetworkActor, MultilayerNetwork


def copy_helper(original_network, copied_network):
    assert id(copied_network) != id(original_network)
    assert len(copied_network) == len(original_network)
    for layer_name in original_network.get_layer_names():
        copied_layer = copied_network[layer_name]
        orig_layer = original_network[layer_name]
        assert id(copied_layer) != id(orig_layer)
        assert nx.utils.graphs_equal(copied_layer, orig_layer) is True


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

    def test_get_layer_names(self):
        """Tests if layer names are read correctly."""
        self.assertEqual(
            self.network.get_layer_names(),
            ["marriage", "business"],
            "Incorrect layer names",
        )

    def test_get_actor(self):
        self.assertEqual(
            self.network.get_actor("Ridolfi"),
            MLNetworkActor("Ridolfi", {"marriage": None}),
        )

    def test_copy(self):
        return copy_helper(self.network, self.network.copy())

    def test___copy__(self):
        return copy_helper(self.network, copy.copy(self.network))

    def test___deepcopy__(self):
        return copy_helper(self.network, copy.deepcopy(self.network))

    def test_is_multiplex_negative(self):
        assert self.network.is_multiplex() is False

    def test_is_multiplex_positive(self):
        assert (
            MultilayerNetwork.from_nx_layer(
                nx.les_miserables_graph(), [1, 2, 3]
            ).is_multiplex()
            is True
        )

    def test_to_multiplex(self):
        multiplexed_net = self.network.to_multiplex()
        all_actors_ids = {a.actor_id for a in multiplexed_net.get_actors()}
        for layer in multiplexed_net.layers:
            assert set(multiplexed_net[layer].nodes) == all_actors_ids


if __name__ == "__main__":
    unittest.main(verbosity=2)
