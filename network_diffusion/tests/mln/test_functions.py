import os
import unittest

import networkx as nx

from network_diffusion import utils
from network_diffusion.mln import MultilayerNetwork
from network_diffusion.mln.centralities import (
    degree,
    multiplexing_coefficient,
    neighbourhood_size,
    torch_voterank_actorwise,
    voterank_actorwise,
)
from network_diffusion.mln.functions import remove_selfloop_edges
from network_diffusion.nets import get_toy_network_piotr

DEGREE_NET_1 = {
    "Ridolfi": 3,
    "Tornabuoni": 4,
    "Strozzi": 4,
    "Peruzzi": 7,
    "Pazzi": 2,
    "Salviati": 3,
    "Medici": 11,
    "Guadagni": 6,
    "Lamberteschi": 5,
    "Castellani": 6,
    "Bischeri": 6,
    "Barbadori": 6,
    "Albizzi": 3,
    "Ginori": 3,
    "Acciaiuoli": 1,
}
DEGREE_NET_2 = {
    1: 3,
    4: 9,
    2: 7,
    3: 6,
    5: 8,
    6: 6,
    7: 4,
    9: 7,
    8: 4,
    10: 5,
    11: 5,
}

NEIGHBOURHOOD_SIZE_NET_1 = {
    "Ridolfi": 3,
    "Tornabuoni": 3,
    "Strozzi": 4,
    "Peruzzi": 5,
    "Pazzi": 2,
    "Salviati": 2,
    "Medici": 8,
    "Guadagni": 4,
    "Lamberteschi": 4,
    "Castellani": 4,
    "Bischeri": 4,
    "Barbadori": 4,
    "Albizzi": 3,
    "Ginori": 3,
    "Acciaiuoli": 1,
}
NEIGHBOURHOOD_SIZE_NET_2 = {
    1: 2,
    4: 5,
    2: 7,
    3: 4,
    5: 4,
    6: 6,
    7: 2,
    9: 4,
    8: 2,
    10: 3,
    11: 3,
}

VOTERANK_NET_1_TOP_5 = [
    "Medici",
    "Peruzzi",
    "Guadagni",
    "Strozzi",
    "Barbadori",
]
VOTERANK_NET_2_TOP_5 = [2, 6, 10, 9, 4]


class TestFunctions(unittest.TestCase):
    """Test functions."""

    def setUp(self):
        """Set up most common testing parameters."""
        self.network_1 = MultilayerNetwork.from_mpx(
            os.path.join(
                utils._get_absolute_path(), "tests/data/florentine.mpx"
            )
        )
        self.network_2 = get_toy_network_piotr()

    def test_compute_multiplexing_coefficient(self):
        """Tests if multiplexing coefficient is computed correctly."""
        names = ["a", "b"]
        network = MultilayerNetwork.from_nx_layer(
            nx.les_miserables_graph(), names
        )
        mcf = multiplexing_coefficient(net=network)
        self.assertEqual(
            mcf,
            1,
            "Multiplexing coefficient for multpilex network should be 1",
        )

        mcf = multiplexing_coefficient(self.network_1)
        self.assertAlmostEqual(
            mcf,
            0.733,
            2,
            'Multiplexing coefficient for "florentine" network should be '
            "almost 0.73",
        )

    def test_degree(self):
        for network, exp_degree in zip(
            [self.network_1, self.network_2], [DEGREE_NET_1, DEGREE_NET_2]
        ):
            raw_degrees = degree(network)
            filtered_degrees = {a.actor_id: d for a, d in raw_degrees.items()}
            self.assertEqual(filtered_degrees, exp_degree)

    def test_neighbourhood_size(self):
        for network, exp_neighbourhood_size in zip(
            [self.network_1, self.network_2],
            [NEIGHBOURHOOD_SIZE_NET_1, NEIGHBOURHOOD_SIZE_NET_2],
        ):
            raw_nss = neighbourhood_size(network)
            filtered_nss = {a.actor_id: d for a, d in raw_nss.items()}
            self.assertEqual(filtered_nss, exp_neighbourhood_size)

    def test_remove_selfloop_edges(self):
        self.network_1["business"].add_edge("Medici", "Medici")
        self.network_1["marriage"].add_edge("Ridolfi", "Ridolfi")

        self.network_1 = remove_selfloop_edges(self.network_1)

        for l_name in self.network_1.layers:
            self.assertEqual(
                list(nx.selfloop_edges(self.network_1[l_name])), []
            )

    def test_voterank_consistency(self):
        """Test that voterank_actorwise and torch_voterank_actorwise produce identical results."""
        for network in [self.network_1, self.network_2]:
            # Test with all actors
            result_cpu = voterank_actorwise(network)
            result_torch = torch_voterank_actorwise(network)
            self.assertEqual(
                [a.actor_id for a in result_cpu],
                [a.actor_id for a in result_torch],
                "VoteRank results should be identical for all actors",
            )
            # Test with limited number of actors
            num_actors = min(5, len(network))
            result_cpu_limited = voterank_actorwise(
                network, number_of_actors=num_actors
            )
            result_torch_limited = torch_voterank_actorwise(
                network, number_of_actors=num_actors
            )
            self.assertEqual(
                [a.actor_id for a in result_cpu_limited],
                [a.actor_id for a in result_torch_limited],
                f"VoteRank results should be identical for {num_actors} actors",
            )

    def test_voterank_actorwise(self):
        """Ensure VoteRank returns expected hardcoded actor sequences."""
        for network, expected in zip(
            [self.network_1, self.network_2],
            [VOTERANK_NET_1_TOP_5, VOTERANK_NET_2_TOP_5],
        ):
            result = voterank_actorwise(network)
            actor_ids = [a.actor_id for a in result[:5]]
            self.assertEqual(
                actor_ids,
                expected,
                "VoteRank output should match the hardcoded sequence",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
