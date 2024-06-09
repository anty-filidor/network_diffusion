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

"""Multilayer Linear Threshold Model class."""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork as MLNetwork
from network_diffusion.models.base_model import BaseModel
from network_diffusion.models.base_model import NetworkUpdateBuffer as NUBuff
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE, NumericType


class MLTModel(BaseModel):
    """
    This model implements Multilayer Linear Threshold Model.

    The model has been presented in paper: "Influence Spread in the
    Heterogeneous Multiplex Linear Threshold Model" by Yaofeng Desmond Zhong,
    Vaibhav Srivastava, and Naomi Ehrich Leonard. This implementation extends
    it to multilayer cases.
    """

    INACTIVE_STATE = "0"
    ACTIVE_STATE = "1"
    PROCESS_NAME = "MLTM"

    def __init__(  # pylint: disable=R0913
        self,
        seeding_budget: Tuple[NumericType, NumericType],
        seed_selector: BaseSeedSelector,
        protocol: str,
        mi_value: float,
    ) -> None:
        """
        Create the object.

        :param seeding_budget: a proportion of INACTIVE and ACTIVE nodes in
            each layer
        :param seed_selector: class that selects initial seeds for simulation
        :param protocol: logical operator that determines how to activate actor
            can be OR (then actor gets activated if it gets positive input in
            one layer) or AND (then actor gets activated if it gets positive
            input in all layers)
        :param mi: activation threshold to transit from INACTIVE to ACTIVE in
            evaluation of the actor in particular layer
        """
        self.__comp_graph = self._create_compartments(seeding_budget, mi_value)
        self.__seed_selector = seed_selector
        if protocol == "AND":
            self.protocol = self._protocol_and
        elif protocol == "OR":
            self.protocol = self._protocol_or
        else:
            raise ValueError("Only OR or AND protocols are allowed!")

    @property
    def _compartmental_graph(self) -> CompartmentalGraph:
        """Compartmental model that defines allowed transitions and states."""
        return self.__comp_graph

    @property
    def _seed_selector(self) -> BaseSeedSelector:
        """A method of selecting seed agents."""
        return self.__seed_selector

    def __str__(self) -> str:
        """Return string representation of the object."""
        descr = f"{BOLD_UNDERLINE}\nMultilayer Linear Threshold Model"
        descr += f"\n{THIN_UNDERLINE}\n"
        descr += self._compartmental_graph._get_desctiprion_str()
        descr += str(self._seed_selector)
        descr += f"{BOLD_UNDERLINE}\nauxiliary parameters\n{THIN_UNDERLINE}"
        descr += f"\n\tprotocol: {self.protocol.__name__}"
        descr += f"\n\tactive state abbreviation: {self.ACTIVE_STATE}"
        descr += f"\n\tinactive state abbreviation: {self.INACTIVE_STATE}"
        descr += f"\n{BOLD_UNDERLINE}"
        return descr

    def _create_compartments(
        self,
        seeding_budget: Tuple[NumericType, NumericType],
        mi_value: float,
    ) -> CompartmentalGraph:
        """
        Create compartmental graph for the model.

        Create one process with two states: 0, 1 and assign transition weight
        equals mi only for transition 0->1.
        """
        compart_graph = CompartmentalGraph()
        assert 0 <= mi_value <= 1, f"incorrect mi value: {mi_value}!"

        # Add allowed states and seeding budget
        compart_graph.add(
            process_name=self.PROCESS_NAME,
            states=[self.INACTIVE_STATE, self.ACTIVE_STATE],
        )
        compart_graph.seeding_budget = {self.PROCESS_NAME: seeding_budget}

        # Add allowed transition
        compart_graph.compile(background_weight=0.0)
        for edge in compart_graph.graph[self.PROCESS_NAME].edges:
            if (
                f"{self.PROCESS_NAME}.{self.INACTIVE_STATE}" in edge[0]
                and f"{self.PROCESS_NAME}.{self.ACTIVE_STATE}" in edge[1]
            ):
                compart_graph.set_transition_canonical(
                    layer=self.PROCESS_NAME,
                    transition=edge,  # type: ignore
                    weight=mi_value,
                )

        return compart_graph

    @staticmethod
    def _protocol_or(inputs: Dict[str, str]) -> bool:
        """Protocol OR for actor activation basing on layer inpulses."""
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.any())

    @staticmethod
    def _protocol_and(inputs: Dict[str, str]) -> bool:
        """Protocol AND for actor activation basing on layer inpulses."""
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.all())

    def determine_initial_states(self, net: MLNetwork) -> List[NUBuff]:
        """
        Determine initial states in the net according to seed selection method.

        :param net: network to initialise seeds for

        :return: a list of nodes with their initial states
        """
        budget = self._compartmental_graph.get_seeding_budget_for_network(
            net=net, actorwise=True
        )
        initial_states: List[NUBuff] = []

        for idx, actor in enumerate(self._seed_selector.actorwise(net=net)):

            # select initial state for given actor according to budget
            if idx < budget[self.PROCESS_NAME][self.ACTIVE_STATE]:
                a_state = self.ACTIVE_STATE
            else:
                a_state = self.INACTIVE_STATE

            # generate update buffer for the actor
            for l_name in actor.layers:
                initial_states.append(
                    NUBuff(
                        node_name=actor.actor_id,
                        layer_name=l_name,
                        new_state=a_state,
                    )
                )

        # return initial states to set in the network
        return initial_states

    def agent_evaluation_step(
        self,
        agent: MLNetworkActor,
        layer_name: str,
        net: MLNetwork,
    ) -> str:
        """
        Try to change state of given actor of the network according to model.

        :param agent: actor to evaluate in given layer
        :param layer_name: a layer where the actor exists
        :param net: a network where the actor exists

        :return: state of the actor in particular layer to be set after epoch
        """
        l_graph: nx.Graph = net.layers[layer_name]

        # trivial case - node is already activated
        current_state = l_graph.nodes[agent.actor_id]["status"]
        if current_state == self.ACTIVE_STATE:
            return current_state

        # import possible transitions for state of the actor
        av_trans = self._compartmental_graph.get_possible_transitions(
            (f"{self.PROCESS_NAME}.{self.INACTIVE_STATE}",), self.PROCESS_NAME
        )

        # iterate through neighbours of node and compute impuls
        impuls = 0
        for neighbour in nx.neighbors(l_graph, agent.actor_id):
            if l_graph.nodes[neighbour]["status"] == self.ACTIVE_STATE:
                impuls += 1 / nx.degree(l_graph, agent.actor_id)  # type:ignore

        # if thresh. has been reached return positive input, otherwise negative
        if impuls > av_trans[self.ACTIVE_STATE]:
            return self.ACTIVE_STATE
        return current_state

    def network_evaluation_step(self, net: MLNetwork) -> List[NUBuff]:
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

            # check if node is already activated, if so don't update it
            if set(actor.states.values()) == set(self.ACTIVE_STATE):
                new_state = self.ACTIVE_STATE

            # try to activate actor in each layer where it exists
            else:
                layer_inputs = {
                    layer: self.agent_evaluation_step(actor, layer, net)
                    for layer in actor.layers
                }
                if self.protocol(layer_inputs):
                    new_state = self.ACTIVE_STATE
                else:
                    new_state = self.INACTIVE_STATE

            activated_nodes.extend(
                [
                    NUBuff(
                        node_name=actor.actor_id,
                        layer_name=layer_name,
                        new_state=new_state,
                    )
                    for layer_name in actor.layers
                ]
            )

        return activated_nodes

    def get_allowed_states(self, net: MLNetwork) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states in each layer of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """
        cmprt = self._compartmental_graph.get_compartments()[self.PROCESS_NAME]
        return {l_name: cmprt for l_name in net.layers}
