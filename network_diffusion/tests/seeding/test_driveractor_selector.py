import unittest

from network_diffusion.nets import get_toy_network_piotr
from network_diffusion.seeding import (
    BetweennessSelector,
    ClosenessSelector,
    DriverActorSelector,
    NeighbourhoodSizeSelector,
)

SUBSELECTORS = (
    (NeighbourhoodSizeSelector(), [2, 4, 3, 9, 10, 6, 5, 11, 1, 7, 8]),
    (BetweennessSelector(), [4, 2, 9, 10, 3, 11, 5, 6, 8, 7, 1]),
    (ClosenessSelector(), [3, 4, 2, 9, 10, 11, 5, 6, 8, 1, 7]),
)


class TestDriverActorSelector(unittest.TestCase):

    def test_nodewise(self):
        selector = NeighbourhoodSizeSelector()
        with self.assertRaises(NotImplementedError):
            selector.nodewise(net=get_toy_network_piotr())

    def test_actorwise(self):
        for subselector, exp_result in SUBSELECTORS:
            with self.subTest(subselector=subselector, exp_result=exp_result):
                das = DriverActorSelector(method=subselector)
                ranking = [
                    a.actor_id for a in das.actorwise(get_toy_network_piotr())
                ]
                print(ranking)
                self.assertEqual(
                    ranking,
                    exp_result,
                    f"Error for method: {subselector.__class__.__name__}",
                )
