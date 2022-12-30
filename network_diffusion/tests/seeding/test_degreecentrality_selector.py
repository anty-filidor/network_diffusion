#!/usr/bin/env python3
# pylint: disable-all
# type: ignore
# flake8: noqa

"""Tests for the network_diffusion.multilayer_network."""

import unittest

import networkx as nx

from network_diffusion import MultilayerNetwork
from network_diffusion.seeding import DegreeCentralitySelector


def create_toy_net() -> MultilayerNetwork:
    layer_1 = nx.Graph(
        (
            [1, 4],
            [2, 4],
            [2, 3],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 5],
            [7, 9],
            [8, 9],
            [8, 10],
        )
    )
    layer_2 = nx.Graph(
        (
            [1, 2],
            [2, 7],
            [2, 11],
            [4, 5],
            [4, 6],
            [5, 11],
            [6, 10],
            [7, 9],
            [8, 9],
            [8, 10],
            [10, 11],
        )
    )
    layer_3 = nx.Graph(
        (
            [1, 4],
            [2, 6],
            [2, 9],
            [3, 4],
            [3, 5],
            [4, 5],
            [5, 6],
            [5, 11],
            [6, 9],
            [7, 9],
            [10, 11],
        )
    )

    return MultilayerNetwork.load_layers_nx(
        [layer_1, layer_2, layer_3], ["l1", "l2", "l3"]
    )


class TestDegreeCentralitySelector(unittest.TestCase):
    """Test DegreeCentralitySelector class."""

    def test_actorwise(self):
        selector = DegreeCentralitySelector()
        obtained_result = selector.actorwise(net=create_toy_net())
        self.assertEqual(
            [obtained_actor.actor_id for obtained_actor in obtained_result],
            [4, 5, 2, 9, 3, 6, 10, 11, 7, 8, 1],
        )

    def test_nodewise(self):
        selector = DegreeCentralitySelector()
        with self.assertRaises(NotImplementedError):
            selector.nodewise(net=create_toy_net())


if __name__ == "__main__":
    unittest.main(verbosity=2)
