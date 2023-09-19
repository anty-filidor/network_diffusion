# Copyright 2023 by Michał Czuba, Piotr Bródka. All Rights Reserved.
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

from network_diffusion.mln.mlnetwork import MultilayerNetwork
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

    def set_initial_states(
        self, net: MultilayerNetwork
    ) -> List[Dict[str, str]]:
        """
        Set initial states in the network according to seed selection method.

        :param net: network to initialise seeds for

        :return: a list of state of the network after initialisation
        """
        if not net.is_multiplex():
            raise ValueError("This model works only with multiplex networks!")

        budget = self._compartmental_graph.get_seeding_budget_for_network(net)
        seed_nodes: List[NetworkUpdateBuffer] = []

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

            # generate update buffer
            for i, _ in enumerate(ranges):
                state = list(l_budget.keys())[i]
                for index in range(ranges[i][0], ranges[i][1]):
                    seed_nodes.append(
                        NetworkUpdateBuffer(
                            node_name=ranking[index],
                            layer_name=l_name,
                            new_state=state,
                        )
                    )

        # set initial states and return json to save in logs
        return self.update_network(net=net, activated_nodes=seed_nodes)

    @staticmethod
    def _find_compartment_state_for_node(
        node: Any, net: MultilayerNetwork
    ) -> Tuple[str, ...]:
        """
        Find proper state in the compartmntal graph for given node.

        :param node: node to find state for
        :param net: a network where node belongs

        :return: a tuple in form on ('process_name.state_name', ...), e.g.
            ('awareness.UA', 'illness.I', 'vaccination.V')
        """
        actor = net.get_actor(node)
        stats = [
            f"{l_name}.{l_state}" for l_name, l_state in actor.states.items()
        ]
        return tuple(sorted(stats))

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
            self._find_compartment_state_for_node(agent, net), layer_name
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
