# Copyright 2023 by Damian Serwata. All Rights Reserved.
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

"""Definition of the temporal network epistemology model."""

from typing import Any, Counter, Dict, List, Tuple

import networkx as nx
import numpy as np

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.base_model import BaseModel
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBff
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE, NumericType


class TemporalNetworkEpistemologyModel(BaseModel):
    """Generalized version of Temporal Network Epistemology Model."""

    PROCESS_NAME = "TNEM"
    A_STATE = "A"
    B_STATE = "B"

    def __init__(
        self,
        seeding_budget: Tuple[NumericType, NumericType],
        seed_selector: BaseSeedSelector,
        trials_nr: int,
        epsilon: float,
    ) -> None:
        """
        Create the object.

        :param seeding_budget: a proportion of INACTIVE and ACTIVE agents
        :param seed_selector: class that selects initial seeds for simulation
        :param trials_nr: number of trials an agent performs when experimenting
            with environment by drawing a sample from the binomial distribution
        :param epsilon: the difference between expected value of B action, and
            A action equal to 0.5
        """
        self.__comp_graph = self._create_compartments(seeding_budget)
        self.__seed_selector = seed_selector
        self.trials_nr = trials_nr
        self.epsilon = epsilon

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
        descr = f"{BOLD_UNDERLINE}\nTemporal Network Epistemology Model"
        descr += f"\n{THIN_UNDERLINE}\n"
        descr += self._compartmental_graph._get_desctiprion_str()
        descr += f"\n{self._seed_selector}"
        descr += f"{BOLD_UNDERLINE}\nauxiliary parameters\n{THIN_UNDERLINE}"
        descr += f"\n{BOLD_UNDERLINE}"
        return descr

    def _create_compartments(
        self, seeding_budget: Tuple[NumericType, NumericType]
    ) -> CompartmentalGraph:
        """
        Create compartmental graph for the model.

        Create one process with two states: A, B.
        """
        compart_graph = CompartmentalGraph()

        # Add allowed states and seeding budget
        compart_graph.add(
            process_name=self.PROCESS_NAME,
            states=[self.A_STATE, self.B_STATE],
        )
        compart_graph.seeding_budget = {self.PROCESS_NAME: seeding_budget}

        # Add allowed transition
        compart_graph.compile()
        return compart_graph

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
            if idx < budget[self.PROCESS_NAME][self.B_STATE]:
                state = self.B_STATE
                belief = np.random.uniform(0.5, 1)
                evidence = np.random.binomial(
                    self.trials_nr, 0.5 + self.epsilon
                )
            else:
                state = self.A_STATE
                belief = np.random.uniform(0, 0.5)
                evidence = 0
            encoded_state = self.encode_actor_status(state, belief, evidence)

            # generate update buffer for the actor
            for l_name in actor.layers:
                seed_nodes.append(
                    NUBff(
                        node_name=actor.actor_id,
                        layer_name=l_name,
                        new_state=encoded_state,
                    )
                )

        # return determined states of nodes in the initial epoch in the network
        return seed_nodes

    @staticmethod
    def encode_actor_status(state: str, belief: float, evidence: int) -> str:
        """
        Encode agent features to str form.

        :param state: state of an actor
        :param belief: level of agent's belief
        :param evidence: nr of successes drawn from binomial distribution in
            an experiment

        :return: a string representation of agent status
        """
        encoded_status = "_".join([state, str(belief), str(evidence)])
        return encoded_status

    @staticmethod
    def decode_actor_status(encoded_status: str) -> Tuple[str, float, int]:
        """
        Decode agent features from str form.

        :param encoded_status: a string representation of agent status

        :return: a tuple with agent state, belief level and evidence
        """
        # TODO: consider operating on dictionary instead of tuple
        state: str = encoded_status.split("_")[0]
        belief: float = float(encoded_status.split("_")[1])
        evidence: int = int(encoded_status.split("_")[2])
        return state, belief, evidence

    def agent_evaluation_step(  # pylint: disable=R0914
        self, agent: MLNetworkActor, layer_name: str, net: MultilayerNetwork
    ) -> str:
        """
        Try to change state of given actor of the network according to model.

        :param agent: actor to evaluate in given layer
        :param net: a network where the actor exists
        :param snapshot_id: currently processed snapshot

        :return: state of the actor to be set in the next snapshot
        """
        l_graph: nx.Graph = net[layer_name]

        # Gather evidence on action B performance
        state, belief, evidence = self.decode_actor_status(
            agent.states[layer_name]
        )
        neighbours_states = [
            self.decode_actor_status(l_graph.nodes[n]["status"])
            for n in l_graph.neighbors(agent.actor_id)
        ]
        b_neighbours_states = [
            state for state in neighbours_states if state[0] == self.B_STATE
        ]
        n = len(b_neighbours_states) * self.trials_nr
        k = np.sum([states[2] for states in b_neighbours_states])
        if state == self.B_STATE:
            n += self.trials_nr
            k += evidence

        # Run credence update
        new_belief = 1 / (
            1
            + (1 - belief)
            * (((0.5 - self.epsilon) / (0.5 + self.epsilon)) ** (2 * k - n))
            / belief
        )

        if new_belief > 0.5:
            new_state = self.B_STATE
            new_evidence = np.random.binomial(
                self.trials_nr, 0.5 + self.epsilon
            )
        else:
            new_state = self.A_STATE
            new_evidence = 0

        encoded_state = self.encode_actor_status(
            new_state, new_belief, new_evidence
        )
        return encoded_state

    def network_evaluation_step(self, net: MultilayerNetwork) -> List[NUBff]:
        """
        Evaluate the given snapshot of the network.

        :param net: a network to evaluate
        :return: list of nodes that changed state after the evaluation
        """
        nodes_to_update: List[NUBff] = []
        for actor in net.get_actors():
            new_state_encoded = self.agent_evaluation_step(
                actor, actor.layers[0], net
            )
            nodes_to_update.extend(
                [
                    NUBff(
                        node_name=actor.actor_id,
                        layer_name=layer_name,
                        new_state=new_state_encoded,
                    )
                    for layer_name in actor.layers
                ]
            )
        return nodes_to_update

    def get_allowed_states(
        self, net: MultilayerNetwork
    ) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """
        cmprt = self._compartmental_graph.get_compartments()[self.PROCESS_NAME]
        return {l_name: cmprt for l_name in net.layers}

    @staticmethod
    def get_states_num(
        net: MultilayerNetwork,
    ) -> Dict[str, Tuple[Tuple[Any, int], ...]]:
        """
        Return states in the network with number of agents that adopted them.

        Vector of states for each agent is following: <state of an actor>
        <agent's belief><evidence>; and we are interested only in the state
        attribute.

        :param net: a network to get states number for
        :return: dictionary with items representing each of layers and with
            summary of nodes states in values
        """
        statistics = {}
        for name, layer in net.layers.items():
            tab = []
            for node in layer.nodes():
                tab.append(layer.nodes[node]["status"].split("_")[0])
            statistics[name] = tuple(Counter(tab).items())
        return statistics
