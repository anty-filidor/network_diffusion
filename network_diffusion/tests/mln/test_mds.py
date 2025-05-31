import unittest

from network_diffusion.mln.mds.greedy_search import get_mds_greedy
from network_diffusion.nets import get_toy_network_piotr


class TestDriverActors(unittest.TestCase):
    def test_compute_driver_actors(self):
        driver_actors = get_mds_greedy(net=get_toy_network_piotr())
        assert {a.actor_id for a in driver_actors} == {2, 3, 4, 8, 9, 11}
