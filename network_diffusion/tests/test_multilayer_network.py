#!/usr/bin/env python3
# pylint: disable-all

"""Tests for the network_diffusion.multilayer_network."""

import os
import unittest

import networkx as nx

from network_diffusion import MultilayerNetwork, utils


class TestMultilayerNetwork(unittest.TestCase):
    """Test MultilayerNetwork class."""

    def setUp(self) -> None:
        """Set up most common testing parameters."""
        self.network = MultilayerNetwork()
        self.network.load_mlx(
            os.path.join(
                utils.get_absolute_path(), "tests/data/florentine.mpx"
            )
        )

    def test_load_mlx(self) -> None:
        """Tests loading network from mpx file."""
        network = MultilayerNetwork()
        network.load_mlx(
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

    def test_load_layers_nx(self) -> None:
        """Tests loading network from several graphs."""
        network = MultilayerNetwork()

        graphs = [nx.random_tree(10)] * 4
        layer_names = ["1", "2", "3", "4"]

        network.load_layers_nx(graphs, layer_names)
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

    def test_load_layer_nx(self) -> None:
        """Checks if creating MLN is correct by load_layer_nx func."""
        network = MultilayerNetwork()
        names = ["a", "b"]
        network.load_layer_nx(nx.les_miserables_graph(), names)

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

    def test_compute_multiplexing_coefficient(self) -> None:
        """Tests if multiplexing coefficient is computed correctly."""
        network = MultilayerNetwork()
        names = ["a", "b"]
        network.load_layer_nx(nx.les_miserables_graph(), names)
        mcf = network._compute_multiplexing_coefficient()
        self.assertEqual(
            mcf,
            1,
            "Multiplexing coefficient for multpilex network should be 1",
        )

        mcf = self.network._compute_multiplexing_coefficient()
        self.assertAlmostEqual(
            mcf,
            0.733,
            2,
            'Multiplexing coefficient for "florentine" network should be '
            "almost 0.73",
        )

    def test_get_nodes_states(self) -> None:
        """Tests if nodes states are being returned correctly."""
        exp_result = {"marriage": ((None, 15),), "business": ((None, 11),)}
        self.assertEqual(
            self.network.get_nodes_states(),
            exp_result,
            "Network states are not same as expected.",
        )

    def test_get_node_state(self) -> None:
        """Tests if node state is being returned correctly."""
        self.assertEqual(
            self.network.get_node_state("Acciaiuoli"),
            ("marriage.None",),
            "Nodes states are not same as expected.",
        )

    def test_get_layer_names(self) -> None:
        """Tests if layer names are read correctly."""
        self.assertEqual(
            self.network.get_layer_names(),
            ["marriage", "business"],
            "Incorrect layer names",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
