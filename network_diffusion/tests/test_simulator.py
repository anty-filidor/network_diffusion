import unittest

import networkx as nx
import numpy as np

from network_diffusion import MultilayerNetwork, Simulator
from network_diffusion.models import DSAAModel
from network_diffusion.tests.models.test_dsaa_model import prepare_compartments


class TestSimulator(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    def setUp(self):
        """Set up most common testing parameters."""
        compartments, phenomena = prepare_compartments()
        self.phenomena = phenomena

        # init multilayer network from nx predefined network
        network = MultilayerNetwork.from_nx_layer(
            nx.les_miserables_graph(), [*phenomena.keys()]
        )
        self.network = network

        # init model
        self.model = DSAAModel(compartments)
        self.model.compartments.seeding_budget = {
            "ill": (84, 13, 3),
            "aware": (77, 23),
            "vacc": (90, 10),
        }

    def test_perform_propagation(self):
        """Check if perform_propagation returns correct values."""
        experiment = Simulator(self.model, self.network)
        initial_states = (
            self.model._compartmental_graph.get_seeding_budget_for_network(
                self.network
            )
        )

        logs = experiment.perform_propagation(50)
        for k, v in logs._global_stats_converted.items():
            for idx, row in v.iterrows():
                if idx == 0:
                    self.assertTrue(
                        np.all(row.values == list(initial_states[k].values())),
                        "Particular states in initial epoch should be equal "
                        "to given during defining an experiment",
                    )
                else:
                    self.assertEqual(
                        row.sum(),
                        np.sum(list(initial_states[k].values())),
                        "Sum of particular node states in epoch must be equal "
                        "to the sum of all nodes in the network's layer "
                        f"sum({row.values}) != {np.sum(initial_states[k])}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
