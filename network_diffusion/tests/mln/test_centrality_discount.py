import unittest

from network_diffusion.mln.centrality_discount import (
    degree_discount,
    degree_discount_networkx,
    neighbourhood_size_discount,
)
from network_diffusion.mln.functions import (
    MultilayerNetwork,
    get_toy_network_cim,
)

DEGREE_DISCOUNT_NETWORKX = {
    "l1_nx": [7, 4, 3, 9, 11, 10, 1, 2, 6, 5, 8, 12],
    "l2_nx": [6, 4, 5, 7, 8, 1, 9, 2, 10, 3, 11, 12],
    "l3_nx": [3, 6, 7, 8, 9, 10, 2, 11, 4, 12, 1, 5],
}
DEGREE_DISCOUNT = {
    "full_net": [
        "7",
        "3",
        "9",
        "10",
        "6",
        "4",
        "8",
        "12",
        "2",
        "1",
        "11",
        "5",
    ],
    "l1": ["7", "4", "3", "9", "11", "10", "1", "2", "6", "5", "8", "12"],
}

NEIGHBOURHOOD_SIZE_DISCOUNT = {
    "full_net": [
        "3",
        "11",
        "7",
        "6",
        "1",
        "8",
        "2",
        "9",
        "4",
        "10",
        "5",
        "12",
    ],
    "l1": ["7", "4", "3", "9", "11", "10", "1", "2", "6", "5", "8", "12"],
}


class TestCentralityDiscount(unittest.TestCase):

    def setUp(self):
        self.networks = {"full_net": get_toy_network_cim()}
        self.networks["l1_nx"] = self.networks["full_net"]["l1"].copy()
        self.networks["l2_nx"] = self.networks["full_net"]["l2"].copy()
        self.networks["l3_nx"] = self.networks["full_net"]["l3"].copy()
        self.networks["l1"] = MultilayerNetwork.from_nx_layer(
            self.networks["full_net"]["l1"].copy(),
            ["l1"],
        )

    def test_degree_discount_networkx(self):
        for net_name, exp_ranking in DEGREE_DISCOUNT_NETWORKX.items():
            ranking = degree_discount_networkx(
                self.networks[net_name], len(exp_ranking)
            )
            self.assertEqual(ranking, exp_ranking)

    def test_degree_discount(self):
        for net_name, exp_ranking in DEGREE_DISCOUNT.items():
            raw_ranking = degree_discount(
                self.networks[net_name], len(exp_ranking)
            )
            filtered_ranking = [str(a.actor_id) for a in raw_ranking]
            self.assertEqual(filtered_ranking, exp_ranking)

    def test_neighbourhood_size_discount(self):
        for net_name, exp_ranking in NEIGHBOURHOOD_SIZE_DISCOUNT.items():
            raw_ranking = neighbourhood_size_discount(
                self.networks[net_name], len(exp_ranking)
            )
            filtered_ranking = [str(a.actor_id) for a in raw_ranking]
            self.assertEqual(filtered_ranking, exp_ranking)


if __name__ == "__main__":
    unittest.main(verbosity=2)
