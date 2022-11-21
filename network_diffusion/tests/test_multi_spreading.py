#!/usr/bin/env python3
# pylint: disable-all
# type: ignore

"""Tests for the network_diffusion.multi_spreading."""

import unittest

import networkx as nx
import numpy as np

from network_diffusion import MultilayerNetwork, MultiSpreading
from network_diffusion.models import DSAAAlgo
from network_diffusion.models.utils.compartmental import CompartmentalGraph


def prepare_data():
    """Set up data needed to perform tests."""
    # init propagation model and set transitions with probabilities
    model = CompartmentalGraph()
    phenomena = {
        "ill": ("S", "I", "R"),
        "aware": ("UA", "A"),
        "vacc": ("UV", "V"),
    }
    for layer, phenomenon in phenomena.items():
        model.add(layer, phenomenon)
    model.compile(background_weight=0)

    # init multilayer network from nx predefined network
    network = MultilayerNetwork()
    network.load_layer_nx(nx.les_miserables_graph(), [*phenomena.keys()])

    model.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.UA"), 0.9)
    model.set_transition_fast("ill.S", "ill.I", ("vacc.V", "aware.A"), 0.05)
    model.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.A"), 0.2)
    model.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.UA"), 0.1)
    model.set_transition_fast("ill.I", "ill.R", ("vacc.V", "aware.A"), 0.7)
    model.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.A"), 0.3)
    model.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.S"), 0.03)
    model.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.I"), 0.01)
    model.set_transition_fast(
        "aware.UA", "aware.A", ("vacc.UV", "ill.S"), 0.05
    )
    model.set_transition_fast("aware.UA", "aware.A", ("vacc.V", "ill.S"), 1)
    model.set_transition_fast("aware.UA", "aware.A", ("vacc.UV", "ill.I"), 0.2)

    return model, phenomena, network


class TestMultiSpreading(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    def setUp(self):
        """Set up most common testing parameters."""
        model, phenomenas, network = prepare_data()
        self.model = DSAAAlgo(model)
        self.phenomenas = phenomenas
        self.network = network

    def test_set_initial_states(self):
        """Check if set_initial_states appends good statuses to nodes."""
        model = self.model
        network = self.network
        experiment = MultiSpreading(model, network)

        phenomenas_num = {
            "ill": (65, 10, 2),
            "aware": (60, 17),
            "vacc": (70, 7),
        }
        experiment.set_initial_states(phenomenas_num)

        # check if number of nodes with certain status is as expected
        for phenomena_name in self.phenomenas.keys():
            phenomena_graph = experiment._network.layers[phenomena_name]
            expected_nodes_states = {
                k: v
                for k, v in zip(
                    self.phenomenas[phenomena_name],
                    phenomenas_num[phenomena_name],
                )
            }
            real_nodes_states = {k: 0 for k in self.phenomenas[phenomena_name]}
            for node in phenomena_graph.nodes:
                status = phenomena_graph.nodes[node]["status"]
                real_nodes_states[status] += 1
            self.assertEqual(
                real_nodes_states,
                expected_nodes_states,
                f"Wrong number of states in layer {phenomena_name}: "
                f"expected {expected_nodes_states} found {real_nodes_states}",
            )

    def test_perform_propagation(self):
        """Check if perform_propagation returns correct values."""
        experiment = MultiSpreading(self.model, self.network)
        initial_states = {
            "ill": (65, 10, 2),
            "aware": (60, 17),
            "vacc": (70, 7),
        }
        experiment.set_initial_states(initial_states)
        logs = experiment.perform_propagation(50)
        for k, v in logs._stats.items():
            for idx, row in v.iterrows():
                if idx == 0:
                    self.assertTrue(
                        np.all(row.values == initial_states[k]),
                        f"Particular states in initial epoch should be equal "
                        f"to given during defining an experiment"
                        f"{row.values} != {initial_states[k]}",
                    )
                else:
                    self.assertEqual(
                        row.sum(),
                        np.sum(initial_states[k]),
                        f"Sum of particular node states in epoch must be equal"
                        f" to the sum of all nodes in the network's layer "
                        f"sum({row.values}) != {np.sum(initial_states[k])}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
