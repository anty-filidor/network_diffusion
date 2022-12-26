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
"""Multiplex Linear Threshold Model class."""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from network_diffusion.models.base_model import BaseModel
from network_diffusion.models.base_model import NetworkUpdateBuffer as NUBuff
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.multilayer_network import MLNetworkActor
from network_diffusion.multilayer_network import MultilayerNetwork as MLNet
from network_diffusion.seeding.base_selector import BaseSeedSelector


class MLTModel(BaseModel):
    """
    This model implements Multiplex Linear Threshold Model.

    The model has been presented in paper: "Influence Spread in the
    Heterogeneous Multiplex Linear Threshold Model" by Yaofeng Desmond Zhong,
    Vaibhav Srivastava, and Naomi Ehrich Leonard.
    """

    INACTIVE_STATE = "0"
    ACTIVE_STATE = "1"

    def __init__(  # pylint: disable=R0913
        self,
        layers: List[str],
        protocol: str,
        seed_selector: BaseSeedSelector,
        seeding_budget: Tuple[int, int],
        mi_value: float,
    ) -> None:
        """
        Create the object.

        :param layers: a list of network layer names to create model for
        :param protocol: logical operator that determines how to activate actor
            can be OR (then actor gets activated if it gets positive input in
            one layer) or AND (then actor gets activated if it gets positive
            input in all layers)
        :param seed_selector: class that selects initial seeds for simulation
        :param seeding_budget: a proportion of INACTIVE and ACTIVE nodes in
            each layer
        :param mi: activation threshold to transit from INACTIVE to ACTIVE in
            evaluation of the actor in particular layer
        """
        compart_graph = self._create_compartments(
            seeding_budget, layers, mi_value
        )
        super().__init__(compart_graph, seed_selector)
        if protocol == "AND":
            self.protocol = self._protocol_and
        elif protocol == "OR":
            self.protocol = self._protocol_or
        else:
            raise ValueError("Only OR or AND protocols are allowed!")

    def _create_compartments(
        self,
        seeding_budget: Tuple[int, int],
        layers: List[str],
        mi_value: float,
    ) -> CompartmentalGraph:
        """
        Create compartmental graph for the model.

        For each process (i.e. layer) create two states: 0, 1 and assign
        transition weight equals mi for each transition 0->1, while for another
        ones 0.
        """
        compart_graph = CompartmentalGraph()
        seeding_budget_full = {}
        assert 0 < mi_value < 1

        # For each process add allowed states and seeding budget
        for layer in layers:
            compart_graph.add(
                layer_name=layer,
                layer_type=[self.INACTIVE_STATE, self.ACTIVE_STATE],
            )
            seeding_budget_full[layer] = seeding_budget

        # Setup seeding budget
        compart_graph.seeding_budget = seeding_budget_full  # type: ignore

        # Add transitions in each layer
        compart_graph.compile(background_weight=0.0)
        for layer in layers:
            for edge in compart_graph.graph[layer].edges:
                if (
                    f"{layer}.{self.INACTIVE_STATE}" in edge[0]
                    and f"{layer}.{self.ACTIVE_STATE}" in edge[1]
                ):
                    compart_graph.set_transition_canonical(
                        layer=layer,
                        transition=edge,  # type: ignore
                        weight=mi_value,
                    )

        return compart_graph

    @staticmethod
    def _protocol_or(inputs: Dict[str, str]) -> bool:
        """Protocol OR for node activation basing on layer inpulses."""
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.any())

    @staticmethod
    def _protocol_and(inputs: Dict[str, str]) -> bool:
        """Protocol OR for node activation basing on layer inpulses."""
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.all())

    def set_initial_states(self, net: MLNet) -> MLNet:
        """
        Set initial states in the network according to seed selection method.

        :param net: network to initialise seeds for
        """
        budget = self._compartmental_graph.get_seeding_budget_for_network(net)
        for l_name, ranking in self._seed_selector(network=net).items():
            budget_active = budget[l_name][self.ACTIVE_STATE]
            nodes_active = ranking[:budget_active]
            l_graph = net.layers[l_name]
            for node in l_graph.nodes():
                if node in nodes_active:
                    l_graph.nodes[node]["status"] = self.ACTIVE_STATE
                else:
                    l_graph.nodes[node]["status"] = self.INACTIVE_STATE
        return net

    def node_evaluation_step(
        self,
        actor_or_node: MLNetworkActor,
        layer_name: str,
        net: MLNet,
    ) -> str:
        """
        Try to change state of given node of the network according to model.

        :param actor: actor to evaluate in given layer
        :param layer_name: a layer where the actor exists
        :param network: a network where the actor exists

        :return: state of the actor in particular layer to be set after epoch
        """
        layer_graph: nx.Graph = net.layers[layer_name]

        # trivial case - node is already activated
        current_state = layer_graph.nodes[actor_or_node.actor_id]["status"]
        if current_state == self.ACTIVE_STATE:
            return current_state

        # import possible transitions for state of the actor
        av_trans = self._compartmental_graph.get_possible_transitions_actor(
            actor_or_node, layer_name
        )

        # iterate through neighbours of node and compute impuls
        impuls = 0
        for neighbour in nx.neighbors(layer_graph, actor_or_node.actor_id):
            if layer_graph.nodes[neighbour]["status"] == self.ACTIVE_STATE:
                impuls += 1 / nx.degree(layer_graph, actor_or_node.actor_id)

        # if thresh. has been reached return positive input, otherwise negative
        if impuls > av_trans[self.ACTIVE_STATE]:
            return self.ACTIVE_STATE
        return current_state

    def network_evaluation_step(self, net: MLNet) -> List[NUBuff]:
        """
        Evaluate the network at one time stamp with MLTModel.

        :param network: a network to evaluate
        :return: list of nodes that changed state after the evaluation
        """
        activated_nodes: List[NUBuff] = []

        # iterate through all actors
        for actor in net.get_actors(shuffle=True):

            # init container for positive inputs in each layer
            layer_inputs = {}

            # try to activate actor in each layer where it exist
            for layer_name in actor.layers:
                layer_inputs[layer_name] = self.node_evaluation_step(
                    actor, layer_name, net
                )

            # apply protocol to determine wether acivate actor or not
            if self.protocol(layer_inputs):
                activated_nodes.extend(
                    [
                        NUBuff(
                            node_name=actor.actor_id,
                            layer_name=layer_name,
                            new_state=self.ACTIVE_STATE,
                        )
                        for layer_name in layer_inputs
                    ]
                )

        return activated_nodes
