import unittest

from network_diffusion.mln.functions import get_toy_network_piotr
from network_diffusion.seeding import (
    BetweennessSelector,
    ClosenessSelector,
    DegreeCentralitySelector,
    DriverActorSelector,
    KatzSelector,
    KShellMLNSeedSelector,
    KShellSeedSelector,
    NeighbourhoodSizeSelector,
    PageRankMLNSeedSelector,
    PageRankSeedSelector,
    RandomSeedSelector,
    VoteRankSeedSelector,
)

SUBSELECTORS = (
    (
        NeighbourhoodSizeSelector(connection_hop=1),
        [2, 4, 3, 9, 11, 8, 6, 5, 10, 1, 7],
    ),
    (
        NeighbourhoodSizeSelector(connection_hop=2),
        [2, 9, 11, 4, 3, 8, 6, 1, 5, 7, 10],
    ),
    (BetweennessSelector(), [11, 4, 2, 9, 8, 3, 5, 6, 10, 7, 1]),
    (ClosenessSelector(), [11, 3, 4, 2, 9, 8, 5, 6, 10, 1, 7]),
    (DegreeCentralitySelector(), [4, 2, 9, 3, 11, 8, 5, 6, 10, 7, 1]),
    (KatzSelector(), [3, 4, 11, 2, 9, 8, 5, 6, 10, 7, 1]),
    (KShellMLNSeedSelector(), [4, 2, 3, 9, 11, 8, 5, 6, 10, 7, 1]),
    (KShellSeedSelector(), [3, 4, 2, 11, 8, 9, 5, 6, 10, 7, 1]),
    (NeighbourhoodSizeSelector(), [2, 4, 3, 9, 11, 8, 6, 5, 10, 1, 7]),
    (PageRankMLNSeedSelector(), [2, 4, 9, 3, 11, 8, 6, 5, 10, 7, 1]),
    (PageRankSeedSelector(), [11, 3, 4, 9, 8, 2, 5, 10, 7, 6, 1]),
    (RandomSeedSelector(), [11, 4, 8, 3, 9, 2, 5, 1, 6, 7, 10]),
    (VoteRankSeedSelector(), [4, 9, 11, 3, 8, 2, 5, 1, 6, 10, 7]),
)


class TestDriverActorSelector(unittest.TestCase):
    def test_actorwise(self):
        selector = NeighbourhoodSizeSelector()
        obtained_result = selector.actorwise(net=get_toy_network_piotr())
        self.assertEqual(
            [obtained_actor.actor_id for obtained_actor in obtained_result],
            [2, 6, 4, 3, 5, 9, 10, 11, 1, 7, 8],
        )

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
