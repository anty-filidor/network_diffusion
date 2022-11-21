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

from typing import List

import networkx as nx
import numpy as np

from network_diffusion.models.base import BaseModel, NetworkUpdateBuffer
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.multilayer_network import MultilayerNetwork


class DSAAAlgo(BaseModel):
    """This model implements algorithm presented at DSAA 2022."""

    def __init__(self, compartmental_graph: CompartmentalGraph) -> None:
        """Create the object."""
        super().__init__(compartmental_graph=compartmental_graph)

    def node_evaluation_step(
        self, node_id: int, layer_name: str, net: MultilayerNetwork
    ) -> str:
        """
        Try to change state of given node of the network according to model.

        :param node_id: id of the node to evaluate
        :param layer_name: a layer where the node exists
        :param network: a network where the node exists

        :return: state of the model after evaluation
        """
        layer_graph: nx.Graph = net.layers[layer_name]
        current_state = layer_graph.nodes[node_id]["status"]

        # import possible transitions for state of the node
        av_trans = self._compartmental_graph.get_possible_transitions(
            net.get_node_state(node_id), layer_name
        )

        # if there is no possible transition don't do anything
        if len(av_trans) <= 0:
            return current_state

        # iterate through neighbours of current node
        for neighbour in nx.neighbors(layer_graph, node_id):
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
            for n in layer_graph.nodes():

                old_state = layer_graph.nodes[n]["status"]
                new_state = self.node_evaluation_step(n, layer_name, net)

                if old_state != new_state:
                    layer_graph.nodes[n]["status"] = new_state

        return activated_nodes
