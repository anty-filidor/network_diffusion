import unittest

import networkx as nx

from network_diffusion.mln import MultilayerNetwork


class TestMLNetworkActor(unittest.TestCase):
    def setUp(self) -> None:
        self.phenomena = {
            "ill": ["S", "I", "R"],
            "aware": ["UA", "A"],
            "vacc": ["UV", "V"],
        }
        self.network = MultilayerNetwork.from_nx_layer(
            nx.les_miserables_graph(), [*self.phenomena.keys()]
        )

    def test_states_as_compartmental_graph(self):
        err_str = "Nodes states are not same as expected."
        actor = "MotherPlutarch"

        self.assertEqual(
            self.network.get_actor(actor).states_as_compartmental_graph(),
            ("aware.None", "ill.None", "vacc.None"),
            err_str,
        )

        self.network.layers["ill"].remove_node("MotherPlutarch")
        self.network.layers["vacc"].nodes["MotherPlutarch"]["status"] = "test"
        self.assertEqual(
            self.network.get_actor(actor).states_as_compartmental_graph(),
            ("aware.None", "vacc.test"),
            err_str,
        )
