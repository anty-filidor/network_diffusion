"""Tests for the network_diffusion.multi_spreading."""
import unittest
from typing import Dict, Tuple

import networkx as nx

from network_diffusion import MultilayerNetwork
from network_diffusion.models import DSAAModel
from network_diffusion.models.utils.compartmental import CompartmentalGraph


def prepare_compartments() -> Tuple[CompartmentalGraph, Dict]:
    """Set up compartments needed to perform tests."""
    cmprt = CompartmentalGraph()
    phenomena = {
        "ill": ["S", "I", "R"],
        "aware": ["UA", "A"],
        "vacc": ["UV", "V"],
    }
    for layer, phenomenon in phenomena.items():
        cmprt.add(layer, phenomenon)
    cmprt.compile(background_weight=0)

    cmprt.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.UA"), 0.9)
    cmprt.set_transition_fast("ill.S", "ill.I", ("vacc.V", "aware.A"), 0.05)
    cmprt.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.A"), 0.2)
    cmprt.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.UA"), 0.1)
    cmprt.set_transition_fast("ill.I", "ill.R", ("vacc.V", "aware.A"), 0.7)
    cmprt.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.A"), 0.3)
    cmprt.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.S"), 0.03)
    cmprt.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.I"), 0.01)
    cmprt.set_transition_fast(
        "aware.UA", "aware.A", ("vacc.UV", "ill.S"), 0.05
    )
    cmprt.set_transition_fast("aware.UA", "aware.A", ("vacc.V", "ill.S"), 1)
    cmprt.set_transition_fast("aware.UA", "aware.A", ("vacc.UV", "ill.I"), 0.2)

    return cmprt, phenomena


class TestDSAAModel(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    # pylint: disable=W0212, C0206, C0201, R0914, R1721

    def setUp(self) -> None:
        """Set up most common testing parameters."""
        compartments, phenomena = prepare_compartments()
        self.phenomena = phenomena

        # init multilayer network from nx predefined network
        network = MultilayerNetwork()
        network.load_layer_nx(nx.les_miserables_graph(), [*phenomena.keys()])
        self.network = network

        # init model
        self.model = DSAAModel(compartments)
        self.model.compartments.seeding_budget = {
            "ill": (84, 13, 3),
            "aware": (77, 23),
            "vacc": (90, 10),
        }
        self.model.set_initial_states(net=self.network)

    def test_set_initial_states(self) -> None:
        """Check if set_initial_states appends good statuses to nodes."""
        numbers_nodes_in_states = (
            self.model._compartmental_graph.get_seeding_budget_for_network(
                self.network
            )
        )
        for phenomena_name in self.phenomena.keys():
            phenomena_graph = self.network.layers[phenomena_name]
            expected_nodes_states = {
                k: v
                for k, v in zip(
                    self.phenomena[phenomena_name],
                    numbers_nodes_in_states[phenomena_name],
                )
            }
            real_nodes_states = {k: 0 for k in self.phenomena[phenomena_name]}

            # obtain real states from the network
            for node in phenomena_graph.nodes:
                status = phenomena_graph.nodes[node]["status"]
                real_nodes_states[status] += 1

            self.assertEqual(
                real_nodes_states,
                expected_nodes_states,
                f"Wrong number of states in layer {phenomena_name}: "
                f"expected {expected_nodes_states} found {real_nodes_states}",
            )
