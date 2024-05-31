import unittest

from network_diffusion.mln.driver_actors import compute_driver_actors
from network_diffusion.mln.functions import get_toy_network_piotr


class TestDriverActors(unittest.TestCase):
    def test_compute_driver_actors(self):
        driver_actors = compute_driver_actors(net=get_toy_network_piotr())
        assert {a.actor_id for a in driver_actors} == {2, 3, 4, 8, 9, 11}
