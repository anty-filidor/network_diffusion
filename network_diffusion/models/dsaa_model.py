# Copyright 2022 by Michał Czuba, Piotr Bródka. All Rights Reserved.
#
# This file is part of Network Diffusion.
#
# Network Diffusion is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Network Diffusion is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# Network Diffusion. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================
"""Functions for the phenomena spreading definition."""

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from network_diffusion.mln.mln_network import MultilayerNetwork
from network_diffusion.models.base_model import BaseModel, NetworkUpdateBuffer
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.seeding.random_selector import RandomSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class DSAAModel(BaseModel):
    """This model implements algorithm presented at DSAA 2022."""

    def __init__(self, compartmental_graph: CompartmentalGraph) -> None:
        """Create the object."""
        super().__init__(
            compartmental_graph=compartmental_graph,
            seed_selector=RandomSeedSelector(),
        )

    def __str__(self) -> str:
        """Return string representation of the object."""
        descr = f"{BOLD_UNDERLINE}\nDSAA Model"
        descr += f"\n{THIN_UNDERLINE}\n"
        descr += self._compartmental_graph.describe()
        descr += str(self._seed_selector)
        descr += f"\n{BOLD_UNDERLINE}"
        return descr

    def set_initial_states(self, net: MultilayerNetwork) -> MultilayerNetwork:
        """
        Set initial states in the network according to seed selection method.

        :param net: network to initialise seeds for
        """
        budget = self._compartmental_graph.get_seeding_budget_for_network(net)

        # set initial states in each layer of network
        for l_name, ranking in self._seed_selector.nodewise(net).items():

            # get data to select seeds in the network
            l_graph = net.layers[l_name]
            l_budget = budget[l_name]
            l_nodes_num = len(l_graph.nodes())

            # set ranges - idk what is it
            _rngs = [
                sum(list(l_budget.values())[:x]) for x in range(len(l_budget))
            ] + [l_nodes_num]
            ranges: List[Tuple[int, int]] = list(zip(_rngs[:-1], _rngs[1:]))

            # append states to nodes
            for i, _ in enumerate(ranges):
                pair = ranges[i]
                state = list(l_budget.keys())[i]
                for index in range(pair[0], pair[1]):
                    l_graph.nodes[ranking[index]]["status"] = state

        return net

    def agent_evaluation_step(
        self, agent: Any, layer_name: str, net: MultilayerNetwork
    ) -> str:
        """
        Try to change state of given node of the network according to model.

        :param agent: id of the node (here agent) to evaluate
        :param layer_name: a layer where the node exists
        :param network: a network where the node exists

        :return: state of the model after evaluation
        """
        layer_graph: nx.Graph = net.layers[layer_name]
        current_state = layer_graph.nodes[agent]["status"]

        # import possible transitions for state of the node
        av_trans = self._compartmental_graph.get_possible_transitions(
            net.get_node_state(agent), layer_name
        )

        # if there is no possible transition don't do anything
        if len(av_trans) <= 0:
            return current_state

        # iterate through neighbours of current node
        for neighbour in nx.neighbors(layer_graph, agent):
            av_new_state = layer_graph.nodes[neighbour]["status"]

            # if state of neighbour node is in possible transitions and
            # tossed 1 due to it's weight
            if (
                av_new_state in av_trans
                and np.random.choice(
                    [0, 1],
                    p=[1 - av_trans[av_new_state], av_trans[av_new_state]],
                )
                == 1
            ):
                # return new state of current node and break iteration
                return av_new_state

        return current_state

    def network_evaluation_step(
        self, net: MultilayerNetwork
    ) -> List[NetworkUpdateBuffer]:
        """
        Evaluate the network at one time stamp according to the model.

        We ae updating nodes 'on the fly', hence the activated_nodes list is
        empty. This behaviour is due to intention to very reflect te algorithm
        presented at DSAA

        :param network: a network to evaluate
        :return: list of nodes that changed state after the evaluation
        """
        activated_nodes: List[NetworkUpdateBuffer] = []

        for layer_name, layer_graph in net.layers.items():
            for node in layer_graph.nodes():

                old_state = layer_graph.nodes[node]["status"]
                new_state = self.agent_evaluation_step(node, layer_name, net)

                if old_state != new_state:
                    layer_graph.nodes[node]["status"] = new_state

        return activated_nodes

    def get_allowed_states(
        self, net: MultilayerNetwork
    ) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states in each layer of net if applied model.

        In this model each process is binded with network's layer, hence we
        return just the compartments and allowed states.

        :param net: a network to determine allowed nodes' states for
        """
        return self._compartmental_graph.get_compartments()
