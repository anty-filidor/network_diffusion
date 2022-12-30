#!/usr/bin/env python3
# pylint: disable-all
# type: ignore

"""Tests for the network_diffusion.multilayer_network."""

import unittest

from network_diffusion.seeding import NeighbourhoodSizeSelector
from network_diffusion.tests.seeding.test_degreecentrality_selector import create_toy_net

class TestNeighbourhoodSizeSelector(unittest.TestCase):
    """Test DegreeCentralitySelector class."""

    def test_actorwise(self):
        selector = NeighbourhoodSizeSelector()
        obtained_result = selector.actorwise(net=create_toy_net())
        self.assertEqual(
            [obtained_actor.actor_id for obtained_actor in obtained_result], 
            [2, 6, 4, 3, 5, 9, 10, 11, 1, 7, 8]
        )

    def test_nodewise(self):
        selector = NeighbourhoodSizeSelector()
        with self.assertRaises(NotImplementedError):
            selector.nodewise(net=create_toy_net())

if __name__ == "__main__":
    unittest.main(verbosity=2)
