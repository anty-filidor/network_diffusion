# Copyright 2023 by Damian Dąbrowski, Michał Czuba. All Rights Reserved.
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

"""Multilayer Independent Cascade Model class."""

import random
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.base_model import BaseModel
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBff
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE, NumericType


class MICModel(BaseModel):
    """This model implements Multilayer Independent Cascade Model."""

    INACTIVE_NODE = "0"
    ACTIVE_NODE = "1"
    ACTIVATED_NODE = "-1"
    PROCESS_NAME = "MICM"

    def __init__(
        self,
        seeding_budget: Tuple[NumericType, NumericType, NumericType],
        seed_selector: BaseSeedSelector,
        protocol: str,
        probability: float,
    ) -> None:
        """
        Create the object.

        :param seeding_budget: a proportion of INACTIVE, ACTIVE and ACTIVATED
            nodes in each layer
        :param seed_selector: class that selects initial seeds for simulation
        :param protocol: logical operator that determines how to activate actor
            can be OR (then actor gets activated if it gets positive input in
            one layer) or AND (then actor gets activated if it gets positive
            input in all layers)
        :param probability: threshold parameter which activate actor(a random
             variable must be greater than this param)
        """
        assert 0 <= probability <= 1, f"incorrect probability: {probability}!"
        self.probability = probability
        self.__comp_graph = self._create_compartments(seeding_budget)
        self.__seed_selector = seed_selector
        if protocol == "AND":
            self.protocol = self._protocol_and
        elif protocol == "OR":
            self.protocol = self._protocol_or
        else:
            raise ValueError("Only AND & OR value is allowed!")

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
        descr = f"{BOLD_UNDERLINE}\nMultilayer Independent Cascade Model"
        descr += f"\n{THIN_UNDERLINE}\n"
        descr += self._compartmental_graph._get_desctiprion_str()
        descr += f"\n{self._seed_selector}"
        descr += f"{BOLD_UNDERLINE}\nauxiliary parameters\n{THIN_UNDERLINE}"
        descr += f"\n\tprotocol: {self.protocol.__name__}"
        descr += f"\n\tinactive state abbreviation: {self.INACTIVE_NODE}"
        descr += f"\n\tactive state abbreviation: {self.ACTIVE_NODE}"
        descr += f"\n\tactivated state abbreviation: {self.ACTIVATED_NODE}"
        descr += f"\n{BOLD_UNDERLINE}"
        return descr

    def _create_compartments(
        self,
        sending_budget: Tuple[NumericType, NumericType, NumericType],
    ) -> CompartmentalGraph:
        """
        Create compartmental graph for the model.

        Create one process with three states: 0, 1, -1 and assign transition
        weights for transition 0 -> 1 (self.probability) and 1 -> -1 (1.0)
        """
        compart_graph = CompartmentalGraph()

        # Add allowed states and seeding budget
        compart_graph.add(
            process_name=self.PROCESS_NAME,
            states=[self.INACTIVE_NODE, self.ACTIVE_NODE, self.ACTIVATED_NODE],
        )
        compart_graph.seeding_budget = {self.PROCESS_NAME: sending_budget}

        # Add allowed transitions
        compart_graph.compile(background_weight=0.0)
        for edge in compart_graph.graph[self.PROCESS_NAME].edges:
            if (
                f"{self.PROCESS_NAME}.{self.INACTIVE_NODE}" in edge[0]
                and f"{self.PROCESS_NAME}.{self.ACTIVE_NODE}" in edge[1]
            ):
                compart_graph.set_transition_canonical(
                    layer=self.PROCESS_NAME,
                    transition=edge,  # type: ignore
                    weight=self.probability,
                )
            elif (
                f"{self.PROCESS_NAME}.{self.ACTIVE_NODE}" in edge[0]
                and f"{self.PROCESS_NAME}.{self.ACTIVATED_NODE}" in edge[1]
            ):
                compart_graph.set_transition_canonical(
                    layer=self.PROCESS_NAME,
                    transition=edge,  # type: ignore
                    weight=1,
                )

        return compart_graph

    @staticmethod
    def _protocol_or(inputs: Dict[str, str]) -> bool:
        """Protocol OR for actor activation basing on layer impulses."""
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.any())

    @staticmethod
    def _protocol_and(inputs: Dict[str, str]) -> bool:
        """Protocol AND for actor activation basing on layer impulses."""
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.all())

    def determine_initial_states(self, net: MultilayerNetwork) -> List[NUBff]:
        """
        Set initial states in the network according to seed selection method.

        :param net: network to initialise seeds for

        :return: a list of state of the network after initialisation
        """
        budget = self._compartmental_graph.get_seeding_budget_for_network(
            net=net, actorwise=True
        )
        seed_nodes: List[NUBff] = []

        for idx, actor in enumerate(self._seed_selector.actorwise(net=net)):

            # select initial state for given actor according to budget
            if idx < budget[self.PROCESS_NAME][self.ACTIVE_NODE]:
                a_state = self.ACTIVE_NODE
            else:
                a_state = self.INACTIVE_NODE

            # generate update buffer for the actor
            for l_name in actor.layers:
                seed_nodes.append(
                    NUBff(
                        node_name=actor.actor_id,
                        layer_name=l_name,
                        new_state=a_state,
                    )
                )

        # set initial states and return json to save in logs
        return seed_nodes

    def agent_evaluation_step(
        self, agent: MLNetworkActor, layer_name: str, net: MultilayerNetwork
    ) -> str:
        """
        Try to change state of given actor of the network according to model.

        :param agent: actor to evaluate in given layer
        :param layer_name: a layer where the actor exists
        :param net: a network where the actor exists

        :return: state of the actor in particular layer to be set after epoch
        """
        l_graph: nx.Graph = net.layers[layer_name]
        current_state = l_graph.nodes[agent.actor_id]["status"]

        # trivial case - node is already active and it looses a potential
        if current_state == self.ACTIVE_NODE:
            return self.ACTIVATED_NODE

        # iterate through neighbours of node and compute impuls
        for neighbour in l_graph.neighbors(agent.actor_id):

            # if neighbour is active, it can send signal to activation of actor
            if l_graph.nodes[neighbour]["status"] == self.ACTIVE_NODE:

                # if a tossed number from unif. distr. > threshold, activ. node
                if random.random() >= self.probability:
                    return self.ACTIVE_NODE

        return current_state

    def network_evaluation_step(self, net: MultilayerNetwork) -> List[NUBff]:
        """
        Evaluate the network at one time stamp with MICModel.

        :param net: a network to evaluate
        :return: list of nodes that changed state after the evaluation
        """
        nodes_to_update: List[NUBff] = []

        for actor in net.get_actors():

            # if actor is already actovated skip its validation
            if list(set(actor.states.values())) == [self.ACTIVATED_NODE]:
                new_state = self.ACTIVATED_NODE

            # otherwise evaluate it on each layer and determine new state
            else:
                inputs = {
                    l_name: self.agent_evaluation_step(actor, l_name, net)
                    for l_name in actor.layers
                }
                if (
                    len(_ := set(inputs.values())) == 1
                    and _.pop() == self.ACTIVATED_NODE
                ):
                    new_state = self.ACTIVATED_NODE
                elif self.protocol(inputs):
                    new_state = self.ACTIVE_NODE
                else:
                    new_state = self.INACTIVE_NODE

            # append calculated state of actor in upcomming epoch to the list
            nodes_to_update.extend(
                [
                    NUBff(
                        node_name=actor.actor_id,
                        layer_name=layer_name,
                        new_state=new_state,
                    )
                    for layer_name in actor.layers
                ]
            )

        return nodes_to_update

    def get_allowed_states(
        self, net: MultilayerNetwork
    ) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states in each layer of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """
        cmprt = self._compartmental_graph.get_compartments()[self.PROCESS_NAME]
        return {l_name: cmprt for l_name in net.layers}
