"""Tests for the network_diffusion.multilayer_network."""

import unittest

from network_diffusion.mln.functions import get_toy_network_piotr
from network_diffusion.seeding import DegreeCentralitySelector


class TestDegreeCentralitySelector(unittest.TestCase):
    """Test DegreeCentralitySelector class."""

    def test_actorwise(self):
        selector = DegreeCentralitySelector()
        obtained_result = selector.actorwise(net=get_toy_network_piotr())
        self.assertEqual(
            [obtained_actor.actor_id for obtained_actor in obtained_result],
            [4, 5, 2, 9, 3, 6, 10, 11, 7, 8, 1],
        )

    def test_nodewise(self):
        selector = DegreeCentralitySelector()
        with self.assertRaises(NotImplementedError):
            selector.nodewise(net=get_toy_network_piotr())


if __name__ == "__main__":
    unittest.main(verbosity=2)
